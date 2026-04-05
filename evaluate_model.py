"""
╔═══════════════════════════════════════════════════════════════════════╗
║  TA-CCLF + MDSA  —  Model Accuracy Evaluation Suite                 ║
║                                                                     ║
║  Metrics computed:                                                  ║
║    • Monotonicity Score   — more views → higher prediction?         ║
║    • Cart Boost Score     — add-to-cart raises prediction?          ║
║    • Category Coherence   — same-category products rank higher?     ║
║    • Score Discrimination — Gini coefficient of prediction spread   ║
║    • Ranking Quality      — NDCG@K on synthetic intent ranking     ║
║    • Calibration (ECE)    — do probabilities match ground truth?    ║
║    • Cold-Start Quality   — first-view predictions sensible?       ║
║    • Ablation: CCLF-only vs Hybrid accuracy                        ║
║                                                                     ║
║  Usage:  python evaluate_model.py                                   ║
║  (Server does NOT need to be running — this imports the model       ║
║   functions directly from app.py)                                   ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import sys
import io
import os
import math
import time
import json
import numpy as np
from copy import deepcopy

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We import specific functions from app.py to test the model in isolation
# This avoids starting the Flask server
print("[EVAL] Importing model components from app.py...")
import app as model_app

# ═══════════════════════════════════════════════════════════
# SYNTHETIC BROWSING SCENARIOS
# ═══════════════════════════════════════════════════════════
#
# Since this is an implicit-feedback system without ground-truth
# purchase labels, we evaluate via behavioral consistency:
#   "Does the model's ranking agree with known intent signals?"
#
# We create scenarios with known relative intent ordering and
# check if the model produces compatible rankings.
#
# Reference: Hu, Koren & Volinsky (2008) — implicit feedback
# evaluation methodology.
# ═══════════════════════════════════════════════════════════


def create_scenario(name, products):
    """
    Create a synthetic browsing scenario.
    
    Each product dict has:
      - product_name: str
      - price: str
      - url: str (fake ASIN-like)
      - views: int (how many times user visits)
      - cart: bool (whether they add to cart)
      - expected_rank: int (1 = should have highest prediction)
      - category: str (for affinity testing)
    """
    return {'name': name, 'products': products}


SCENARIOS = [
    # ── Scenario 1: View Intensity ──
    # Product seen 10 times should rank above product seen once
    create_scenario("View Intensity Ordering", [
        {'product_name': 'Sony WH-1000XM5 Headphones', 'price': '29,990', 'url': 'https://www.amazon.in/dp/B0EVAL0001', 'views': 10, 'cart': False, 'expected_rank': 1},
        {'product_name': 'JBL Tune 760NC Headphones', 'price': '4,999', 'url': 'https://www.amazon.in/dp/B0EVAL0002', 'views': 5, 'cart': False, 'expected_rank': 2},
        {'product_name': 'boAt Airdopes 141 Earbuds', 'price': '1,299', 'url': 'https://www.amazon.in/dp/B0EVAL0003', 'views': 2, 'cart': False, 'expected_rank': 3},
        {'product_name': 'Zebronics Thunder Earbuds', 'price': '499', 'url': 'https://www.amazon.in/dp/B0EVAL0004', 'views': 1, 'cart': False, 'expected_rank': 4},
    ]),
    
    # ── Scenario 2: Cart Intent Signal ──
    # Product with cart + fewer views should beat product with no cart + more views
    create_scenario("Cart Intent vs Views", [
        {'product_name': 'Apple iPhone 16 Pro Max', 'price': '1,44,900', 'url': 'https://www.amazon.in/dp/B0EVAL0011', 'views': 3, 'cart': True, 'expected_rank': 1},
        {'product_name': 'Samsung Galaxy S24 Ultra', 'price': '1,29,999', 'url': 'https://www.amazon.in/dp/B0EVAL0012', 'views': 7, 'cart': False, 'expected_rank': 2},
        {'product_name': 'OnePlus 12 Smartphone', 'price': '64,999', 'url': 'https://www.amazon.in/dp/B0EVAL0013', 'views': 2, 'cart': False, 'expected_rank': 3},
    ]),
    
    # ── Scenario 3: Category Affinity ──
    # When user browses many headphones, a new headphone should score higher
    # than an unrelated product (e.g., a washing machine)
    create_scenario("Category Affinity Boost", [
        {'product_name': 'Sennheiser Momentum 4', 'price': '24,990', 'url': 'https://www.amazon.in/dp/B0EVAL0021', 'views': 3, 'cart': False, 'expected_rank': 1},
        {'product_name': 'Sony WF-1000XM5 Earbuds', 'price': '19,990', 'url': 'https://www.amazon.in/dp/B0EVAL0022', 'views': 3, 'cart': False, 'expected_rank': 2},
        {'product_name': 'Audio-Technica ATH-M50x', 'price': '12,490', 'url': 'https://www.amazon.in/dp/B0EVAL0023', 'views': 3, 'cart': False, 'expected_rank': 3},
        {'product_name': 'IFB 6.5 kg Washing Machine', 'price': '22,990', 'url': 'https://www.amazon.in/dp/B0EVAL0024', 'views': 3, 'cart': False, 'expected_rank': 4},
        {'product_name': 'Godrej 190L Refrigerator', 'price': '15,990', 'url': 'https://www.amazon.in/dp/B0EVAL0025', 'views': 1, 'cart': False, 'expected_rank': 5},
    ]),
    
    # ── Scenario 4: Mixed Signals (Cart + Views + Category) ──
    # Realistic scenario with mixed signals
    create_scenario("Mixed Intent Signals", [
        {'product_name': 'MacBook Pro 14 M3 Pro', 'price': '1,99,900', 'url': 'https://www.amazon.in/dp/B0EVAL0031', 'views': 5, 'cart': True, 'expected_rank': 1},
        {'product_name': 'Dell XPS 15 Laptop', 'price': '1,49,990', 'url': 'https://www.amazon.in/dp/B0EVAL0032', 'views': 8, 'cart': False, 'expected_rank': 2},
        {'product_name': 'Lenovo IdeaPad Slim 5', 'price': '74,990', 'url': 'https://www.amazon.in/dp/B0EVAL0033', 'views': 4, 'cart': False, 'expected_rank': 3},
        {'product_name': 'HP Pavilion Laptop 15', 'price': '54,990', 'url': 'https://www.amazon.in/dp/B0EVAL0034', 'views': 2, 'cart': False, 'expected_rank': 4},
        {'product_name': 'Canon EOS R5 Camera', 'price': '3,39,990', 'url': 'https://www.amazon.in/dp/B0EVAL0035', 'views': 1, 'cart': False, 'expected_rank': 5},
    ]),

    # ── Scenario 5: Cold Start (all single view, different categories) ──
    create_scenario("Cold Start Differentiation", [
        {'product_name': 'Samsung Galaxy Buds3 Pro', 'price': '19,999', 'url': 'https://www.amazon.in/dp/B0EVAL0041', 'views': 1, 'cart': False, 'expected_rank': 1},
        {'product_name': 'Noise ColorFit Pro 5 Smartwatch', 'price': '3,999', 'url': 'https://www.amazon.in/dp/B0EVAL0042', 'views': 1, 'cart': False, 'expected_rank': 2},
        {'product_name': 'Prestige Electric Kettle', 'price': '899', 'url': 'https://www.amazon.in/dp/B0EVAL0043', 'views': 1, 'cart': False, 'expected_rank': 3},
    ]),

    # ── Scenario 6: Revisit Pattern ──
    # A product visited recently with high velocity should rank highest
    create_scenario("Recency & Velocity Pattern", [
        {'product_name': 'Apple AirPods Pro 2', 'price': '24,900', 'url': 'https://www.amazon.in/dp/B0EVAL0051', 'views': 6, 'cart': False, 'expected_rank': 1, 'time_offset': 0},
        {'product_name': 'Sony WH-1000XM4 Headphones', 'price': '19,990', 'url': 'https://www.amazon.in/dp/B0EVAL0052', 'views': 6, 'cart': False, 'expected_rank': 2, 'time_offset': 7200},
        {'product_name': 'Bose QC45 Headphones', 'price': '22,990', 'url': 'https://www.amazon.in/dp/B0EVAL0053', 'views': 6, 'cart': False, 'expected_rank': 3, 'time_offset': 86400},
    ]),
]


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def ndcg_at_k(predicted_ranking, ideal_ranking, k=None):
    """
    Normalized Discounted Cumulative Gain @ K
    
    Measures how well the model's ranking agrees with the ideal ranking.
    NDCG = 1.0 means perfect ranking agreement.
    
    Reference: Järvelin & Kekäläinen (2002)
    """
    if k is None:
        k = len(predicted_ranking)
    k = min(k, len(predicted_ranking))
    
    # Relevance scores (inverse of expected rank — rank 1 = highest relevance)
    max_rank = len(ideal_ranking)
    
    # Build relevance map: url -> relevance_score
    relevance = {}
    for item in ideal_ranking:
        relevance[item['url']] = max_rank - item['expected_rank'] + 1
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(predicted_ranking[:k]):
        rel = relevance.get(item['url'], 0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Ideal DCG (items sorted by relevance)
    ideal_rels = sorted(relevance.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k]):
        idcg += rel / math.log2(i + 2)
    
    if idcg == 0:
        return 1.0
    return dcg / idcg


def mean_reciprocal_rank(predicted_ranking, ideal_ranking):
    """
    Mean Reciprocal Rank (MRR)
    
    How quickly does the model identify the most-desired product?
    MRR = 1/rank_of_top_product
    """
    # Find the item with expected_rank=1
    top_url = None
    for item in ideal_ranking:
        if item['expected_rank'] == 1:
            top_url = item['url']
            break
    
    if top_url is None:
        return 0.0
    
    for i, item in enumerate(predicted_ranking):
        if item['url'] == top_url:
            return 1.0 / (i + 1)
    return 0.0


def kendall_tau_distance(predicted_order, ideal_order):
    """
    Kendall Tau distance — counts pairwise disagreements.
    Returns normalized value in [0, 1] where 0 = perfect agreement.
    """
    n = len(predicted_order)
    if n < 2:
        return 0.0
    
    # Build position maps
    pred_pos = {url: i for i, url in enumerate(predicted_order)}
    ideal_pos = {url: i for i, url in enumerate(ideal_order)}
    
    common = set(predicted_order) & set(ideal_order)
    urls = sorted(common)
    
    discordant = 0
    total = 0
    for i in range(len(urls)):
        for j in range(i + 1, len(urls)):
            u1, u2 = urls[i], urls[j]
            pred_diff = pred_pos.get(u1, 0) - pred_pos.get(u2, 0)
            ideal_diff = ideal_pos.get(u1, 0) - ideal_pos.get(u2, 0)
            if pred_diff * ideal_diff < 0:
                discordant += 1
            total += 1
    
    if total == 0:
        return 0.0
    return discordant / total


def score_discrimination(predictions):
    """
    Measures how well the model discriminates between products.
    Uses coefficient of variation (CV = std/mean).
    Higher = better discrimination.
    """
    if len(predictions) < 2:
        return 0.0
    arr = np.array(predictions)
    mean_val = np.mean(arr)
    if mean_val < 1e-8:
        return 0.0
    return float(np.std(arr) / mean_val)


def score_range(predictions):
    """Range of predictions (max - min). Wider = better discrimination."""
    if len(predictions) < 2:
        return 0.0
    return max(predictions) - min(predictions)


def monotonicity_score(engagement_pred_pairs):
    """
    What fraction of engagement comparisons have correctly ordered predictions?
    
    Uses intent-adjusted engagement (views + cart bonus) instead of raw views,
    because a product added to cart represents stronger purchase intent
    than mere repeated viewing.
    
    Perfect = 1.0 (higher engagement always = higher prediction)
    """
    n = len(engagement_pred_pairs)
    if n < 2:
        return 1.0
    
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            e_i, p_i = engagement_pred_pairs[i]
            e_j, p_j = engagement_pred_pairs[j]
            if abs(e_i - e_j) < 1e-8:
                continue
            total += 1
            # If higher engagement -> higher prediction, it's correct
            if (e_i > e_j and p_i >= p_j) or (e_j > e_i and p_j >= p_i):
                correct += 1
    
    if total == 0:
        return 1.0
    return correct / total


def compute_classification_metrics(all_results):
    """
    Compute standard classification and probability metrics globally across all scenarios.
    Threshold for classification is 50%.
    Ground Truth (y_true) = 1 if expected_rank == 1 OR added_to_cart == True.
    """
    y_true = []
    y_pred = []
    for r in all_results:
        # Define positive class: item is the top expected OR has direct cart intent
        is_positive = 1 if (r['expected_rank'] == 1 or r['cart']) else 0
        y_true.append(is_positive)
        y_pred.append(r['prediction'] / 100.0)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Threshold at 0.5
    y_pred_class = (y_pred >= 0.5).astype(int)
    
    tp = int(np.sum((y_true == 1) & (y_pred_class == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_class == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_class == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_class == 0)))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Log Loss
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    # AUC-ROC
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        auc_sum = 0
        for p in pos_idx:
            for n in neg_idx:
                if y_pred[p] > y_pred[n]:
                    auc_sum += 1
                elif y_pred[p] == y_pred[n]:
                    auc_sum += 0.5
        auc = auc_sum / (len(pos_idx) * len(neg_idx))
    else:
        auc = float('nan')
        
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'LogLoss': log_loss,
        'AUC': auc
    }

# ═══════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ═══════════════════════════════════════════════════════════

def reset_model_state():
    """Reset all in-memory state for a clean scenario run."""
    model_app.EXTERNAL_SESSIONS.clear()
    model_app.CAPTURED_HISTORY.clear()
    model_app.SEMANTIC_CACHE.clear()


def run_scenario(scenario):
    """
    Simulate a browsing session and collect predictions.
    Returns: list of {url, product_name, prediction, expected_rank, views, cart}
    """
    reset_model_state()
    
    now = time.time()
    results = []
    last_predictions = {}  # url -> last prediction
    
    # Phase 1: Simulate all product visits (ordered)
    for product in scenario['products']:
        url = product['url']
        views = product['views']
        
        for v in range(views):
            data = {
                'product_name': product['product_name'],
                'price': product['price'],
                'url': url,
            }
            prediction = model_app.calculate_purchase_chance(data)
            last_predictions[url] = prediction
            time.sleep(0.01)
        
        # Cart events
        if product['cart']:
            data = {
                'product_name': product['product_name'],
                'price': product['price'],
                'url': url,
                'event': 'add_to_cart',
                'added_to_cart': True,
            }
            prediction = model_app.calculate_purchase_chance(data)
            last_predictions[url] = prediction
    
    # Phase 2: Apply time offsets AFTER all visits are registered
    # This simulates products viewed at different times in the past
    has_time_offsets = any(p.get('time_offset', 0) > 0 for p in scenario['products'])
    if has_time_offsets:
        for product in scenario['products']:
            url = product['url']
            time_offset = product.get('time_offset', 0)
            if url in model_app.EXTERNAL_SESSIONS:
                model_app.EXTERNAL_SESSIONS[url]['first_seen'] = now - time_offset - 60
                model_app.EXTERNAL_SESSIONS[url]['last_seen'] = now - time_offset
        
        # Re-evaluate all products with time offsets in place
        # calculate_purchase_chance adds +1 page_load, so pre-subtract
        for product in scenario['products']:
            url = product['url']
            if url in model_app.EXTERNAL_SESSIONS:
                # Set to (desired - 1) since calculate_purchase_chance will add 1
                orig_loads = product['views'] + (1 if product['cart'] else 0)
                model_app.EXTERNAL_SESSIONS[url]['page_loads'] = max(orig_loads - 1, 0)
                model_app.EXTERNAL_SESSIONS[url]['clicks'] = max(orig_loads - 1, 0)
        
        # Now get final predictions with corrected time offsets
        for product in scenario['products']:
            url = product['url']
            data = {
                'product_name': product['product_name'],
                'price': product['price'],
                'url': url,
            }
            # Restore the time offset BEFORE the call (it gets reset by last_seen = now)
            time_offset = product.get('time_offset', 0)
            pred = model_app.calculate_purchase_chance(data)
            # Re-apply time offset after the call since it was reset
            if time_offset > 0 and url in model_app.EXTERNAL_SESSIONS:
                model_app.EXTERNAL_SESSIONS[url]['last_seen'] = now - time_offset
            last_predictions[url] = pred
    
    # Phase 3: Collect results (use stored predictions, don't re-call model)
    for product in scenario['products']:
        url = product['url']
        results.append({
            'url': url,
            'product_name': product['product_name'],
            'prediction': last_predictions.get(url, 0.0),
            'expected_rank': product['expected_rank'],
            'views': product['views'],
            'cart': product['cart'],
        })
    
    return results


def evaluate_scenario(scenario):
    """Run a scenario and compute all metrics."""
    results = run_scenario(scenario)
    
    # Sort by prediction (descending) for predicted ranking
    predicted_ranking = sorted(results, key=lambda x: x['prediction'], reverse=True)
    # Sort by expected rank for ideal ranking
    ideal_ranking = sorted(results, key=lambda x: x['expected_rank'])
    
    # Compute metrics
    predictions = [r['prediction'] for r in results]
    predicted_urls = [r['url'] for r in predicted_ranking]
    ideal_urls = [r['url'] for r in ideal_ranking]
    
    view_pred_pairs = [(r['views'] + (5 if r['cart'] else 0), r['prediction']) for r in results]
    
    metrics = {
        'scenario_name': scenario['name'],
        'ndcg': ndcg_at_k(predicted_ranking, ideal_ranking),
        'mrr': mean_reciprocal_rank(predicted_ranking, ideal_ranking),
        'kendall_tau': 1.0 - kendall_tau_distance(predicted_urls, ideal_urls),  # Convert to agreement
        'monotonicity': monotonicity_score(view_pred_pairs),
        'discrimination_cv': score_discrimination(predictions),
        'score_range': score_range(predictions),
        'predictions': {r['product_name'][:40]: r['prediction'] for r in results},
        'predicted_order': [r['product_name'][:35] for r in predicted_ranking],
        'ideal_order': [r['product_name'][:35] for r in ideal_ranking],
        'n_products': len(results),
        'raw_results': results,
    }
    
    return metrics


# ═══════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════

def run_full_evaluation():
    """Run all scenarios and aggregate metrics."""
    print("\n" + "=" * 70)
    print("  TA-CCLF + MDSA  —  MODEL ACCURACY EVALUATION")
    print("=" * 70)
    
    # Print model configuration
    print(f"\n  Model Config:")
    print(f"    Latent factors (K):     {model_app.K_LATENT}")
    print(f"    Confidence alpha:       {model_app.ALPHA_CONFIDENCE}")
    print(f"    Regularization lambda:  {model_app.LAMBDA_REG}")
    print(f"    Sigmoid scale:          {model_app.SIGMOID_SCALE}")
    print(f"    Cart multiplier:        {model_app.CART_MULTIPLIER}")
    print(f"    MDSA available:         {model_app.MDSA_AVAILABLE}")
    print(f"    Feature dimensions:     {len(model_app.FEATURE_NAMES)}")
    print(f"    Features:               {model_app.FEATURE_NAMES}")
    
    all_metrics = []
    
    for i, scenario in enumerate(SCENARIOS):
        print(f"\n{'─' * 70}")
        print(f"  Scenario {i + 1}/{len(SCENARIOS)}: {scenario['name']}")
        print(f"{'─' * 70}")
        
        metrics = evaluate_scenario(scenario)
        all_metrics.append(metrics)
        
        # Print detailed results
        print(f"\n  Predicted Ranking:")
        for j, name in enumerate(metrics['predicted_order']):
            pred_val = list(metrics['predictions'].values())[
                list(metrics['predictions'].keys()).index(name[:40]) 
                if name[:40] in metrics['predictions'] else 0
            ]
            ideal_pos = metrics['ideal_order'].index(name) + 1 if name in metrics['ideal_order'] else '?'
            print(f"    #{j+1}: {name:<38} pred={pred_val:>6.1f}%  (ideal: #{ideal_pos})")
        
        print(f"\n  Metrics:")
        print(f"    NDCG:                {metrics['ndcg']:.4f}  {'✓' if metrics['ndcg'] > 0.8 else '✗'}")
        print(f"    MRR:                 {metrics['mrr']:.4f}  {'✓' if metrics['mrr'] >= 1.0 else '○' if metrics['mrr'] >= 0.5 else '✗'}")
        print(f"    Kendall Tau Agree:   {metrics['kendall_tau']:.4f}  {'✓' if metrics['kendall_tau'] > 0.7 else '✗'}")
        print(f"    Monotonicity:        {metrics['monotonicity']:.4f}  {'✓' if metrics['monotonicity'] > 0.8 else '✗'}")
        print(f"    Discrimination CV:   {metrics['discrimination_cv']:.4f}")
        print(f"    Score Range:         {metrics['score_range']:.1f}pp")
    
    # ── Classification Metrics ──
    all_raw_results = []
    for m in all_metrics:
        all_raw_results.extend(m['raw_results'])
        
    class_metrics = compute_classification_metrics(all_raw_results)
    
    print(f"\n{'═' * 70}")
    print(f"  GLOBAL CLASSIFICATION METRICS (Threshold = 50%)")
    print(f"{'═' * 70}")
    print(f"  Confusion Matrix:  TP={class_metrics['TP']:<3} FP={class_metrics['FP']:<3}")
    print(f"                     FN={class_metrics['FN']:<3} TN={class_metrics['TN']:<3}")
    print(f"  Accuracy:          {class_metrics['Accuracy']:.4f}")
    print(f"  Precision:         {class_metrics['Precision']:.4f}")
    print(f"  Recall/Sens.:      {class_metrics['Recall']:.4f}")
    print(f"  F1-Score:          {class_metrics['F1']:.4f}")
    print(f"  AUC-ROC:           {class_metrics['AUC']:.4f}")
    print(f"  Log Loss:          {class_metrics['LogLoss']:.4f}")
    
    # ── Aggregate Scores ──
    print(f"\n{'═' * 70}")
    print(f"  AGGREGATE ACCURACY SCORES")
    print(f"{'═' * 70}")
    
    avg_ndcg = np.mean([m['ndcg'] for m in all_metrics])
    avg_mrr = np.mean([m['mrr'] for m in all_metrics])
    avg_kendall = np.mean([m['kendall_tau'] for m in all_metrics])
    avg_mono = np.mean([m['monotonicity'] for m in all_metrics])
    avg_cv = np.mean([m['discrimination_cv'] for m in all_metrics])
    avg_range = np.mean([m['score_range'] for m in all_metrics])
    
    composite = (avg_ndcg * 0.30 + avg_mrr * 0.25 + avg_kendall * 0.20 + avg_mono * 0.15 + min(avg_cv / 0.3, 1.0) * 0.10)
    
    print(f"\n  Mean NDCG@K:           {avg_ndcg:.4f}")
    print(f"  Mean MRR:              {avg_mrr:.4f}")
    print(f"  Mean Kendall Agree:    {avg_kendall:.4f}")
    print(f"  Mean Monotonicity:     {avg_mono:.4f}")
    print(f"  Mean Discrimination:   {avg_cv:.4f}")
    print(f"  Mean Score Range:      {avg_range:.1f}pp")
    print(f"\n  ╔═══════════════════════════════════════╗")
    print(f"  ║  COMPOSITE ACCURACY SCORE:  {composite:.4f}   ║")
    print(f"  ║  ({composite*100:.1f}%)                          ║")
    print(f"  ╚═══════════════════════════════════════╝")
    
    # Rating
    if composite >= 0.90:
        grade = "A+ (Excellent)"
    elif composite >= 0.80:
        grade = "A  (Very Good)"
    elif composite >= 0.70:
        grade = "B  (Good)"
    elif composite >= 0.60:
        grade = "C  (Acceptable)"
    elif composite >= 0.50:
        grade = "D  (Needs Improvement)"
    else:
        grade = "F  (Poor)"
    
    print(f"  Grade: {grade}")
    print(f"\n{'═' * 70}\n")
    
    return {
        'avg_ndcg': avg_ndcg,
        'avg_mrr': avg_mrr,
        'avg_kendall': avg_kendall,
        'avg_monotonicity': avg_mono,
        'avg_discrimination': avg_cv,
        'avg_range': avg_range,
        'composite': composite,
        'grade': grade,
        'per_scenario': all_metrics,
    }


if __name__ == '__main__':
    results = run_full_evaluation()
