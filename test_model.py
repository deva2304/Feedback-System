"""Test script: verifies deduplication, view counting, and cart boosting."""
import urllib.request
import json
import time

SERVER = 'http://127.0.0.1:5000'

def post(endpoint, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f'{SERVER}{endpoint}', data=body, headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

def get(endpoint):
    r = urllib.request.urlopen(f'{SERVER}{endpoint}')
    return json.loads(r.read())

print("=" * 60)
print("TEST 1: Product deduplication with different URL params")
print("=" * 60)

# Visit 1: First time seeing the product
r1 = post('/api/external/track', {
    'product_name': 'Test Widget Pro 2026',
    'price': '4,999',
    'url': 'https://www.amazon.in/dp/B0TESTAAAA/ref=sr_1_1?keywords=test'
})
print(f"Visit 1: prediction={r1['prediction']}%, views={r1['page_loads']}, revisit={r1.get('is_revisit', False)}")

time.sleep(0.5)

# Visit 2: Same ASIN, different URL params (like ?th=1)
r2 = post('/api/external/track', {
    'product_name': 'Test Widget Pro 2026',
    'price': '4,999',
    'url': 'https://www.amazon.in/dp/B0TESTAAAA?th=1'
})
print(f"Visit 2: prediction={r2['prediction']}%, views={r2['page_loads']}, revisit={r2.get('is_revisit', False)}")

time.sleep(0.5)

# Visit 3: Same ASIN again
r3 = post('/api/external/track', {
    'product_name': 'Test Widget Pro 2026',
    'price': '4,999',
    'url': 'https://www.amazon.in/dp/B0TESTAAAA/ref=pd_sbs?th=1&psc=1'
})
print(f"Visit 3: prediction={r3['prediction']}%, views={r3['page_loads']}, revisit={r3.get('is_revisit', False)}")

# Check history for duplicates
history = get('/api/external/history')
test_entries = [e for e in history if 'Test Widget' in e.get('product_name', '')]
print(f"\nHistory entries for 'Test Widget': {len(test_entries)} (should be 1)")
if test_entries:
    e = test_entries[0]
    print(f"  Views: {e['page_loads']} (should be 3)")
    print(f"  Prediction: {e['prediction']}%")
    print(f"  URL: {e['url']}")

assert len(test_entries) == 1, f"FAIL: Expected 1 entry, got {len(test_entries)}"
assert test_entries[0]['page_loads'] == 3, f"FAIL: Expected 3 views, got {test_entries[0]['page_loads']}"
print("\n>>> TEST 1 PASSED: No duplicates + view count incremented!")

print("\n" + "=" * 60)
print("TEST 2: Prediction increases with more views")
print("=" * 60)
assert r2['prediction'] >= r1['prediction'], f"FAIL: prediction didn't increase: {r1['prediction']} -> {r2['prediction']}"
assert r3['prediction'] >= r2['prediction'], f"FAIL: prediction didn't increase: {r2['prediction']} -> {r3['prediction']}"
print(f"  View 1: {r1['prediction']}%")
print(f"  View 2: {r2['prediction']}%  (+{round(r2['prediction']-r1['prediction'],1)}%)")
print(f"  View 3: {r3['prediction']}%  (+{round(r3['prediction']-r2['prediction'],1)}%)")
print(">>> TEST 2 PASSED: More views = higher prediction!")

print("\n" + "=" * 60)
print("TEST 3: Add-to-cart boosts prediction")
print("=" * 60)
pre_cart = r3['prediction']

r_cart = post('/api/external/add-to-cart', {
    'product_name': 'Test Widget Pro 2026',
    'price': '4,999',
    'url': 'https://www.amazon.in/dp/B0TESTAAAA',
    'event': 'add_to_cart'
})
print(f"  Before cart: {pre_cart}%")
print(f"  After cart:  {r_cart['prediction']}%  (+{round(r_cart['prediction']-pre_cart,1)}%)")
assert r_cart['prediction'] > pre_cart, f"FAIL: Cart didn't boost prediction"
print(">>> TEST 3 PASSED: Add-to-cart significantly boosts prediction!")

print("\n" + "=" * 60)
print("TEST 4: Model status shows CCLF")
print("=" * 60)
status = get('/api/status')
print(f"  Model: {status['model']}")
print(f"  Features: {status['features']}")
print(f"  Latent factors: {status['latent_factors']}")
print(f"  Explained variance: {status['explained_variance']}%")
assert 'Correlation' in status['model'], "FAIL: Model name doesn't mention Correlation"
assert status['latent_factors'] == 5, "FAIL: Wrong number of latent factors"
print(">>> TEST 4 PASSED: CCLF model is active!")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)

# Cleanup: remove test product from history
# (won't affect CSV but that's fine for testing)
