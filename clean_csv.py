"""One-time script to deduplicate captured_data.csv using ASIN-based URL normalization."""
import csv
import re
import os

def normalize_amazon_url(url):
    if not url:
        return url
    asin_match = re.search(r'/dp/([A-Z0-9]{10})', url, re.IGNORECASE)
    if asin_match:
        asin = asin_match.group(1).upper()
        domain = 'www.amazon.in'
        domain_match = re.search(r'(www\.amazon\.[a-z.]+)', url)
        if domain_match:
            domain = domain_match.group(1)
        return f'https://{domain}/dp/{asin}'
    return url.split('?')[0].split('#')[0]

CSV_FILE = 'captured_data.csv'

rows = []
seen = {}
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        url = normalize_amazon_url(row.get('url', ''))
        if url in seen:
            existing = rows[seen[url]]
            existing_pred = float(existing.get('prediction', 0))
            new_pred = float(row.get('prediction', 0))
            merged_loads = int(existing.get('page_loads', 1)) + int(row.get('page_loads', 1))
            if new_pred > existing_pred:
                row['url'] = url
                row['page_loads'] = str(merged_loads)
                rows[seen[url]] = row
            else:
                existing['page_loads'] = str(merged_loads)
                existing['timestamp'] = row.get('timestamp', existing['timestamp'])
        else:
            row['url'] = url
            seen[url] = len(rows)
            rows.append(row)

with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Cleaned CSV: {len(rows)} unique products")
for r in rows:
    name = r.get('product_name', 'Unknown')[:60]
    loads = r.get('page_loads', '1')
    url_short = r.get('url', '')[:55]
    print(f"  {name:60s} | Views: {loads} | {url_short}")
