from flask import Flask, render_template, request, jsonify
import os
import json
import csv
import re
import time as _time
import math
from datetime import datetime
import random
import numpy as np

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response


# ═══════════════════════════════════════════════════════════
# REAL-TIME DATA STORAGE
# ═══════════════════════════════════════════════════════════

CAPTURED_CSV = 'captured_data.csv'
CAPTURED_HISTORY = []  # In-memory list for the dashboard (last 20)
EXTERNAL_SESSIONS = {}  # Session tracking for prediction logic


# ═══════════════════════════════════════════════════════════
# URL NORMALIZATION — ASIN-BASED DEDUPLICATION
# ═══════════════════════════════════════════════════════════

def normalize_amazon_url(url):
    """
    Extract the canonical Amazon product URL based on ASIN.
    
    Amazon URLs contain tracking params, referral codes, and variant selectors
    like ?th=1 that cause the same product to appear as different URLs.
    
    This extracts just the ASIN (10-char alphanumeric ID) and returns a
    canonical form: https://www.amazon.in/dp/ASIN
    
    Examples:
      /dp/B0FQF5DG3P?th=1  →  https://www.amazon.in/dp/B0FQF5DG3P
      /dp/B0FQF5DG3P/ref=sr_1_1_sspa?crid=...  →  same
    """
    if not url:
        return url
    
    # Extract ASIN from /dp/XXXXXXXXXX pattern
    asin_match = re.search(r'/dp/([A-Z0-9]{10})', url, re.IGNORECASE)
    if asin_match:
        asin = asin_match.group(1).upper()
        # Detect domain
        domain = 'www.amazon.in'
        domain_match = re.search(r'(www\.amazon\.[a-z.]+)', url)
        if domain_match:
            domain = domain_match.group(1)
        return f'https://{domain}/dp/{asin}'
    
    # Fallback: strip query parameters
    return url.split('?')[0].split('#')[0]


# ═══════════════════════════════════════════════════════════
# PRODUCT CATEGORY DETECTION
# ═══════════════════════════════════════════════════════════

CATEGORY_KEYWORDS = {
    'Speakers': ['speaker', 'loudspeaker', 'subwoofer', 'soundbar', 'woofer', 'tweeter', 'amplifier', 'audio monitor', 'studio monitor', 'home theatre', 'home theater', 'boombox', 'thump', 'mackie', 'jbl', 'marshall'],
    'Headphones': ['headphone', 'earphone', 'earbud', 'earbuds', 'headset', 'airpod', 'in-ear', 'over-ear', 'on-ear', 'neckband', 'wireless buds', 'tws'],
    'Laptops': ['laptop', 'notebook', 'macbook', 'chromebook', 'ultrabook', 'thinkpad', 'ideapad', 'vivobook', 'zenbook', 'surface pro'],
    'Mobiles': ['smartphone', 'mobile phone', 'iphone', 'galaxy s', 'pixel phone', 'oneplus', 'redmi', 'realme', 'poco', 'nothing phone', 'motorola', 'vivo', 'oppo', 'samsung galaxy'],
    'Tablets': ['tablet', 'ipad', 'galaxy tab', 'kindle', 'fire hd', 'lenovo tab'],
    'Cameras': ['camera', 'dslr', 'mirrorless', 'gopro', 'canon eos', 'nikon', 'sony alpha', 'webcam', 'action cam', 'camcorder'],
    'TVs & Displays': ['television', 'smart tv', ' tv ', 'monitor', 'led tv', 'oled', 'qled', '4k tv', 'display', 'projector'],
    'Gaming': ['gaming', 'xbox', 'playstation', 'ps5', 'ps4', 'nintendo', 'controller', 'joystick', 'gaming mouse', 'gaming keyboard', 'console'],
    'Wearables': ['smartwatch', 'smart watch', 'fitness band', 'fitness tracker', 'apple watch', 'galaxy watch', 'garmin', 'fitbit'],
    'Storage': ['hard drive', 'ssd', 'pen drive', 'flash drive', 'memory card', 'micro sd', 'external drive', 'nas'],
    'Keyboards & Mice': ['keyboard', 'mouse', 'mechanical keyboard', 'trackpad', 'ergonomic mouse'],
    'Networking': ['router', 'modem', 'wifi', 'wi-fi', 'range extender', 'mesh', 'ethernet', 'switch'],
    'Home Appliances': ['refrigerator', 'washing machine', 'microwave', 'air conditioner', 'vacuum', 'purifier', 'cooler', 'heater', 'iron', 'mixer', 'grinder', 'blender', 'oven', 'dishwasher', 'geyser', 'water heater'],
    'Fashion': ['shirt', 'shoes', 'sneaker', 'watch', 'bag', 'backpack', 'jacket', 'jeans', 'dress', 'kurta', 'saree', 'sunglasses', 'wallet', 'belt'],
    'Books': ['book', 'novel', 'textbook', 'paperback', 'hardcover', 'kindle edition', 'ebook'],
    'Beauty': ['perfume', 'moisturizer', 'shampoo', 'conditioner', 'serum', 'face wash', 'sunscreen', 'lipstick', 'makeup', 'grooming', 'trimmer', 'razor'],
    'Sports & Fitness': ['dumbbell', 'treadmill', 'yoga mat', 'cricket', 'football', 'badminton', 'gym', 'protein', 'bicycle', 'cycle'],
}

CATEGORY_ICONS = {
    'Speakers': '🔊', 'Headphones': '🎧', 'Laptops': '💻', 'Mobiles': '📱',
    'Tablets': '📋', 'Cameras': '📷', 'TVs & Displays': '📺', 'Gaming': '🎮',
    'Wearables': '⌚', 'Storage': '💾', 'Keyboards & Mice': '⌨️',
    'Networking': '🌐', 'Home Appliances': '🏠', 'Fashion': '👕',
    'Books': '📚', 'Beauty': '💄', 'Sports & Fitness': '🏋️', 'Other': '📦',
}


def categorize_product(product_name):
    """Auto-detect product category from its name using keyword matching."""
    name_lower = product_name.lower()
    best_match = 'Other'
    best_count = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw.lower() in name_lower)
        if match_count > best_count:
            best_count = match_count
            best_match = category
    return best_match


# Load any previously saved data on startup
def load_captured_history():
    """Load previously captured interactions from CSV on server start (deduplicated by normalized URL)."""
    if not os.path.exists(CAPTURED_CSV):
        return
    try:
        with open(CAPTURED_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            seen_urls = {}  # normalized_url -> index in CAPTURED_HISTORY
            for row in reader:
                raw_url = row.get('url', '')
                url = normalize_amazon_url(raw_url)
                product_name = row.get('product_name', 'Unknown')
                category = row.get('category', '') or categorize_product(product_name)
                entry = {
                    'product_name': product_name,
                    'price': row.get('price', 'N/A'),
                    'url': url,
                    'prediction': float(row.get('prediction', 0)),
                    'timestamp': float(row.get('timestamp', 0)),
                    'category': category,
                    'category_icon': CATEGORY_ICONS.get(category, '📦'),
                    'added_to_cart': row.get('added_to_cart', 'false').lower() == 'true',
                    'page_loads': int(row.get('page_loads', 1)),
                }
                if url and url in seen_urls:
                    # Merge: keep latest timestamp, sum page_loads, keep highest prediction
                    idx = seen_urls[url]
                    existing = CAPTURED_HISTORY[idx]
                    existing['page_loads'] = existing['page_loads'] + entry['page_loads']
                    existing['timestamp'] = max(existing['timestamp'], entry['timestamp'])
                    existing['prediction'] = max(existing['prediction'], entry['prediction'])
                    if entry['added_to_cart']:
                        existing['added_to_cart'] = True
                else:
                    seen_urls[url] = len(CAPTURED_HISTORY)
                    CAPTURED_HISTORY.append(entry)
        # Keep only last 20 unique products
        while len(CAPTURED_HISTORY) > 20:
            CAPTURED_HISTORY.pop(0)
        
        # Rebuild sessions from loaded history
        for entry in CAPTURED_HISTORY:
            url = entry['url']
            EXTERNAL_SESSIONS[url] = {
                'clicks': entry['page_loads'],
                'first_seen': entry['timestamp'] - 60,  # approximate
                'last_seen': entry['timestamp'],
                'added_to_cart': entry.get('added_to_cart', False),
                'page_loads': entry['page_loads'],
            }
        
        print(f"[TRACKER] Loaded {len(CAPTURED_HISTORY)} unique products from {CAPTURED_CSV}")
    except Exception as e:
        print(f"[TRACKER] Could not load history: {e}")


def save_to_csv(data, prediction, category='Other', added_to_cart=False, page_loads=1):
    """Save captured interaction to CSV. Updates existing product row (by normalized URL) instead of duplicating."""
    raw_url = data.get('url', '')
    url = normalize_amazon_url(raw_url)
    new_row = {
        'timestamp': str(_time.time()),
        'product_name': data.get('product_name', 'Unknown'),
        'price': data.get('price', 'N/A'),
        'url': url,
        'prediction': str(prediction),
        'category': category,
        'added_to_cart': str(added_to_cart),
        'page_loads': str(page_loads),
        'captured_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    fieldnames = ['timestamp', 'product_name', 'price', 'url', 'prediction',
                  'category', 'added_to_cart', 'page_loads', 'captured_at']

    try:
        rows = []
        updated = False
        if os.path.exists(CAPTURED_CSV):
            with open(CAPTURED_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_url = normalize_amazon_url(row.get('url', ''))
                    if existing_url == url and url:
                        rows.append(new_row)
                        updated = True
                    else:
                        rows.append(row)
        if not updated:
            rows.append(new_row)

        with open(CAPTURED_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"[TRACKER] CSV save error: {e}")


# Load history on startup
load_captured_history()


# ═══════════════════════════════════════════════════════════
# CORRELATION-CONSTRAINED LATENT FACTOR MODEL (CCLF)
# ═══════════════════════════════════════════════════════════
#
#  This is the core prediction engine for purchase probability.
#
#  Instead of using simple heuristic scoring, we build a proper
#  Correlation-Constrained Latent Factor Model that:
#
#  1. Extracts behavioral feature vectors for each product interaction
#  2. Builds a Pearson correlation matrix across features
#  3. Applies SVD decomposition to discover latent factors
#  4. Uses correlation constraints to penalize redundant features
#  5. Computes purchase probability via sigmoid-calibrated dot product
#
#  Mathematical formulation:
#    R ≈ P × Σ × Q^T    (SVD of correlation-constrained feature matrix)
#    score = Σ (p_u · σ_k · q_i) for k latent factors
#    P(purchase) = sigmoid(α · score + β · cart_boost + γ · view_intensity)
#
# ═══════════════════════════════════════════════════════════

# Model hyperparameters
K_LATENT = 5            # Number of latent factors
ALPHA_CONFIDENCE = 40   # Confidence scaling: c = 1 + α·r
LAMBDA_REG = 0.1        # Regularization strength
SIGMOID_SCALE = 2.8     # Sigmoid steepness
SIGMOID_SHIFT = -1.2    # Sigmoid horizontal shift
CART_MULTIPLIER = 1.65  # How much add-to-cart boosts the score
VIEW_LOG_BASE = 1.8     # Logarithmic view scaling base


def extract_features(session, data, all_sessions):
    """
    Extract an 8-dimensional behavioral feature vector for a product interaction.
    
    Features:
      f1: view_count       - Number of times product page was loaded (log-scaled)
      f2: session_duration  - Total time browsing this product (normalized)
      f3: price_signal      - Inverse price signal (cheaper items score higher)
      f4: recency           - How recently the product was last viewed (exponential decay)
      f5: view_velocity     - Views per minute (engagement intensity)
      f6: cart_signal       - Binary: 1 if added to cart, 0 otherwise
      f7: category_affinity - How many products in the same category were viewed
      f8: relative_interest - This product's views relative to average views across all products
    """
    now = _time.time()
    page_loads = session.get('page_loads', 1)
    first_seen = session.get('first_seen', now)
    last_seen = session.get('last_seen', now)
    duration = max(last_seen - first_seen, 1)
    
    # f1: View count (log-scaled to avoid dominance)
    view_count = math.log(1 + page_loads) / math.log(1 + 20)  # Normalize to ~[0,1] (20 views ≈ 1.0)
    
    # f2: Session duration (sigmoid-normalized)
    session_duration = 1 - math.exp(-duration / 300)  # 5 minutes → ~0.63
    
    # f3: Price signal (inverse — cheaper items get higher signal)
    price_str = str(data.get('price', '0')).replace(',', '').replace('₹', '').replace('$', '').replace('.', '').strip()
    try:
        price_val = float(price_str) if price_str else 0
    except ValueError:
        price_val = 0
    if price_val > 0:
        price_signal = math.exp(-price_val / 100000)  # ₹1,00,000 → ~0.37
    else:
        price_signal = 0.5
    
    # f4: Recency (exponential decay — recent views score higher)
    time_since_last = now - last_seen
    recency = math.exp(-time_since_last / 3600)  # 1 hour → ~0.37
    
    # f5: View velocity (views per minute — measures engagement intensity)
    if duration > 0:
        view_velocity = min((page_loads / (duration / 60)), 5) / 5  # Cap at 5 views/min, normalize
    else:
        view_velocity = 0.5
    
    # f6: Cart signal (strongest buying intent indicator)
    cart_signal = 1.0 if session.get('added_to_cart', False) else 0.0
    
    # f7: Category affinity (how many products in same category?)
    product_category = categorize_product(data.get('product_name', ''))
    same_cat_count = sum(1 for s_url, s in all_sessions.items() 
                         if s.get('category', '') == product_category and s_url != normalize_amazon_url(data.get('url', '')))
    category_affinity = min(same_cat_count / 5, 1.0)  # 5+ same-category → maxed out
    
    # f8: Relative interest (this product's views vs avg views)
    all_loads = [s.get('page_loads', 1) for s in all_sessions.values()]
    avg_loads = sum(all_loads) / max(len(all_loads), 1)
    if avg_loads > 0:
        relative_interest = min(page_loads / (avg_loads * 2), 1.0)
    else:
        relative_interest = 0.5
    
    return np.array([
        view_count, session_duration, price_signal, recency,
        view_velocity, cart_signal, category_affinity, relative_interest
    ])

FEATURE_NAMES = [
    'view_count', 'session_duration', 'price_signal', 'recency',
    'view_velocity', 'cart_signal', 'category_affinity', 'relative_interest'
]


def build_correlation_matrix(feature_vectors, time_weights=None):
    """
    Build a Time-Decayed Pearson correlation matrix with Tikhonov Regularization.
    C_decay = X^T W X + λI
    
    Returns: (n_features × n_features) correlation matrix
    """
    if len(feature_vectors) < 2:
        # Not enough data for meaningful correlation — use identity
        n = feature_vectors[0].shape[0] if feature_vectors else 8
        return np.eye(n)
    
    # Stack into matrix (n_products × n_features)
    X = np.array(feature_vectors)
    
    # Center the features
    X_centered = X - X.mean(axis=0)
    
    # Compute correlation matrix with stability guard
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero
    X_normalized = X_centered / std
    
    # Time-decay weighting (Ebbinghaus)
    if time_weights is not None:
        W = np.diag(time_weights)
    else:
        W = np.eye(X.shape[0])
    
    n = X.shape[0]
    corr_matrix = (X_normalized.T @ W @ X_normalized) / max(n - 1, 1)
    
    # Tikhonov Regularization (L2) to guarantee numerical stability
    corr_matrix += LAMBDA_REG * np.eye(corr_matrix.shape[0])
    
    # Clamp to [-1, 1] for numerical stability
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    return corr_matrix


def compute_latent_factors(corr_matrix, k=K_LATENT):
    """
    Decompose the correlation matrix using SVD to extract latent factors.
    
    R = U × Σ × V^T
    
    The top-k singular values and vectors capture the dominant 
    patterns of correlated behavior. These are the 'latent factors'
    that explain why users who view products repeatedly tend to buy.
    
    Returns: (U, sigma, Vt) truncated to k factors
    """
    try:
        U, sigma, Vt = np.linalg.svd(corr_matrix, full_matrices=False)
        
        # Truncate to k factors
        k = min(k, len(sigma))
        U_k = U[:, :k]
        sigma_k = sigma[:k]
        Vt_k = Vt[:k, :]
        
        return U_k, sigma_k, Vt_k
    except np.linalg.LinAlgError:
        # Fallback if SVD fails
        n = corr_matrix.shape[0]
        k = min(k, n)
        return np.eye(n, k), np.ones(k), np.eye(k, n)


def compute_explained_variance(sigma):
    """Calculate the explained variance ratio from singular values."""
    total = np.sum(sigma ** 2)
    if total < 1e-10:
        return 0.0
    return float(np.sum(sigma[:K_LATENT] ** 2) / total * 100)


def calculate_purchase_chance(data):
    """
    Time-Aware Correlation-Constrained Latent Factor Model (TA-CCLF)
    
    Academic Implementation featuring:
    1. Ebbinghaus Time Decay for novelty scaling
    2. Tikhonov (L2) Regularized SVD inversion
    3. Implicit Feedback Confidence Matrix (Hu, Koren & Volinsky, 2008)
    4. Temperature-calibrated Platt Scaling probability mapping
    
    Mathematical formulation:
      Feature vector:  x_i ∈ R^8
      Time Weights:    W = diag(e^{-t / \lambda_{decay}})
      Correlation:     C_{reg} = X^T W X / (n-1) + \lambda I
      SVD:             C_{reg} = U Σ V^T
      Confidence:      c_{uj} = 1 + \alpha_{views} + \beta_{cart} + \gamma_{cat}
      User latent:     p_u = \frac{\sum (c_{uj} \cdot U^T x_j)}{\sum c_{uj}}
      Item latent:     q_i = U^T x_i
      Constrained:     q_adj = \Sigma^{-1/2} q_i
      Raw score:       s = p_u · diag(Σ) · q_adj
      Final Prob:      P(y=1) = \sigma(\frac{s - \mu}{\tau})   (Platt scale)
    """
    raw_url = data.get('url', 'unknown')
    url = normalize_amazon_url(raw_url)
    now = _time.time()

    # Initialize or update session
    if url not in EXTERNAL_SESSIONS:
        EXTERNAL_SESSIONS[url] = {
            'clicks': 0, 'first_seen': now, 'last_seen': now,
            'added_to_cart': False, 'page_loads': 0,
            'category': categorize_product(data.get('product_name', '')),
        }

    session = EXTERNAL_SESSIONS[url]
    session['clicks'] += 1
    session['page_loads'] = session.get('page_loads', 0) + 1
    session['last_seen'] = now
    
    # Store category in session for cross-product affinity
    if not session.get('category'):
        session['category'] = categorize_product(data.get('product_name', ''))

    # Check if add_to_cart was signaled
    if data.get('event') == 'add_to_cart' or data.get('added_to_cart'):
        session['added_to_cart'] = True

    # ── Step 1: Extract feature vector for this product ──
    feature_vec = extract_features(session, data, EXTERNAL_SESSIONS)
    
    # ── Step 2: Collect feature vectors from ALL tracked products ──
    all_feature_vecs = []
    all_urls = []
    for s_url, s_session in EXTERNAL_SESSIONS.items():
        s_data = {
            'product_name': '',
            'price': '0',
            'url': s_url,
        }
        # Find product info from history
        for h in CAPTURED_HISTORY:
            if h.get('url') == s_url:
                s_data['product_name'] = h.get('product_name', '')
                s_data['price'] = h.get('price', '0')
                break
        fv = extract_features(s_session, s_data, EXTERNAL_SESSIONS)
        all_feature_vecs.append(fv)
        all_urls.append(s_url)
    
    # Ensure current product's data is included
    if url not in all_urls:
        all_feature_vecs.append(feature_vec)
        all_urls.append(url)
    else:
        # Update the feature vector for this URL
        idx = all_urls.index(url)
        all_feature_vecs[idx] = feature_vec
    
    # ── Step 3: Build Time-Aware Correlation Matrix with Tikhonov Regularization ──
    # Ebbinghaus Decay weighting
    time_weights = []
    for s_url in all_urls:
        s_time = EXTERNAL_SESSIONS.get(s_url, {}).get('last_seen', now)
        decay = math.exp(-(now - s_time) / 86400) # 1 day half-life
        time_weights.append(decay)
        
    corr_matrix = build_correlation_matrix(all_feature_vecs, time_weights=time_weights)
    
    # ── Step 4: SVD decomposition for latent factors ──
    U_k, sigma_k, Vt_k = compute_latent_factors(corr_matrix)
    
    # ── Step 5: Project feature vector into latent space ──
    item_latent = U_k.T @ feature_vec[:U_k.shape[0]]
    
    # ── Step 6: Compute User Latent Vector via Implicit Confidence (Hu, Koren, Volinsky) ──
    user_latent = np.zeros(len(sigma_k))
    total_weight = 0
    
    for i, s_url in enumerate(all_urls):
        s = EXTERNAL_SESSIONS.get(s_url, {})
        
        # c_ui = 1 + α*views + β*cart + γ*affinity
        # This replaces the ad-hoc post-SVD boosts with mathematical confidence mapping.
        c_ui = 1.0 
        c_ui += ALPHA_CONFIDENCE * math.log(1 + s.get('page_loads', 1))
        c_ui += 60.0 if s.get('added_to_cart') else 0.0
        c_ui += 15.0 if s.get('category') == session.get('category') else 0.0
        
        fv = all_feature_vecs[i]
        proj = U_k.T @ fv[:U_k.shape[0]]
        
        user_latent += c_ui * proj
        total_weight += c_ui
    
    # Soft-normalization (Confidence Damping)
    # Instead of dividing by total_weight (which cancels out confidence for single items),
    # we divide by log(2 + total_weight) so that higher absolute engagement = larger intent vector.
    if total_weight > 0:
        user_latent /= math.log(2 + total_weight)
    
    # ── Step 7: Correlation-constrained adjustment ──
    try:
        eigenvalues = np.maximum(sigma_k, 0.01)
        constraint_weights = 1.0 / np.sqrt(eigenvalues)
        constraint_weights /= np.max(constraint_weights)  # Normalize
    except Exception:
        constraint_weights = np.ones(len(sigma_k))
    
    # ── Step 8: Compute constrained dot product score ──
    raw_score = np.sum(user_latent * sigma_k * item_latent * constraint_weights)
    
    # ── Step 9: Platt Scaling for Probability Output ──
    # P(purchase) = sigmoid((s - mu) / tau)
    tau_temp = 1.6   # Temperature parameter
    mu_shift = -2.2  # Mean threshold shift
    
    logit = (raw_score - mu_shift) / tau_temp
    # Cap logit to prevent overflow
    logit_capped = max(min(logit, 10), -10)
    
    probability = 1.0 / (1.0 + math.exp(-logit_capped))
    
    # Scale to percentage with realistic bounds
    purchase_pct = probability * 100
    
    # Apply correlation-based confidence adjustment
    # More data = more confident = wider range of predictions
    n_products = len(EXTERNAL_SESSIONS)
    confidence_factor = min(n_products / 5, 1.0)  # Full confidence at 5+ products
    
    # Blend toward base rate for low confidence
    base_rate = 8.0  # Base purchase probability with minimal data
    purchase_pct = base_rate + (purchase_pct - base_rate) * (0.4 + 0.6 * confidence_factor)
    
    # Clamp to realistic bounds
    purchase_pct = max(3.0, min(purchase_pct, 97.0))
    
    return round(purchase_pct, 1)


def get_model_meta():
    """Get current model metadata for display."""
    all_feature_vecs = []
    for s_url, s_session in EXTERNAL_SESSIONS.items():
        s_data = {'product_name': '', 'price': '0', 'url': s_url}
        for h in CAPTURED_HISTORY:
            if h.get('url') == s_url:
                s_data['product_name'] = h.get('product_name', '')
                s_data['price'] = h.get('price', '0')
                break
        fv = extract_features(s_session, s_data, EXTERNAL_SESSIONS)
        all_feature_vecs.append(fv)
    
    if len(all_feature_vecs) >= 2:
        corr_matrix = build_correlation_matrix(all_feature_vecs)
        _, sigma, _ = compute_latent_factors(corr_matrix)
        explained_var = compute_explained_variance(sigma)
    else:
        explained_var = 0.0
    
    return {
        'type': 'Time-Aware CCLF (TA-CCLF) with Tikhonov Regularization',
        'signals': FEATURE_NAMES,
        'k_latent': K_LATENT,
        'explained_variance': round(explained_var, 1),
        'n_products': len(EXTERNAL_SESSIONS),
        'data_source': 'Live Amazon Browsing (via Browser Extension)',
        'storage': CAPTURED_CSV,
        'total_captured': len(CAPTURED_HISTORY),
    }


# ═══════════════════════════════════════════════════════════
# ANALYTICS — Computed from YOUR real captured data
# ═══════════════════════════════════════════════════════════

def compute_live_analytics():
    """Generate analytics from captured real-time data."""
    if not CAPTURED_HISTORY:
        return {
            'total_products_viewed': 0,
            'total_interactions': 0,
            'avg_prediction': 0,
            'highest_prediction': 0,
            'unique_products': 0,
            'products': [],
            'categories': {},
            'cart_count': 0,
            'total_views': 0,
        }

    predictions = [h['prediction'] for h in CAPTURED_HISTORY]
    unique_names = set(h['product_name'] for h in CAPTURED_HISTORY)
    total_views = sum(h.get('page_loads', 1) for h in CAPTURED_HISTORY)

    # Category breakdown
    categories = {}
    cart_count = 0
    for h in CAPTURED_HISTORY:
        cat = h.get('category', 'Other')
        if cat not in categories:
            categories[cat] = {'count': 0, 'icon': CATEGORY_ICONS.get(cat, '📦'), 'products': []}
        categories[cat]['count'] += 1
        categories[cat]['products'].append(h['product_name'])
        if h.get('added_to_cart'):
            cart_count += 1

    return {
        'total_products_viewed': len(CAPTURED_HISTORY),
        'total_interactions': total_views,
        'avg_prediction': round(sum(predictions) / len(predictions), 1),
        'highest_prediction': round(max(predictions), 1),
        'unique_products': len(unique_names),
        'products': CAPTURED_HISTORY[:10],
        'categories': categories,
        'cart_count': cart_count,
        'total_views': total_views,
    }


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def home():
    stats = compute_live_analytics()
    return render_template('index.html',
                           total_products=stats['total_products_viewed'],
                           total_interactions=stats['total_interactions'],
                           avg_prediction=stats['avg_prediction'],
                           highest_prediction=stats['highest_prediction'],
                           unique_products=stats['unique_products'])


@app.route('/analytics')
def analytics():
    stats = compute_live_analytics()
    return render_template('visuals.html', stats=stats)


@app.route('/about')
def about():
    return render_template('about.html', model_meta=get_model_meta())


# ═══════════════════════════════════════════════════════════
# REAL-TIME TRACKING API (BROWSER EXTENSION)
# ═══════════════════════════════════════════════════════════

@app.route('/api/external/track', methods=['POST', 'OPTIONS'])
def track_external():
    """Receive tracking data from the Amazon Safe Tracker extension."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        product_name = data.get('product_name', 'Unknown Product')
        price = data.get('price', 'N/A')
        raw_url = data.get('url', 'unknown')
        url = normalize_amazon_url(raw_url)
        category = categorize_product(product_name)
        category_icon = CATEGORY_ICONS.get(category, '📦')
        
        # Normalize the URL in the data dict so the model uses canonical URLs
        data['url'] = url

        # Check if this is a revisit
        is_revisit = url in EXTERNAL_SESSIONS
        old_page_loads = EXTERNAL_SESSIONS.get(url, {}).get('page_loads', 0) if is_revisit else 0

        # Calculate prediction (this also updates the session)
        chance = calculate_purchase_chance(data)
        
        # Get session info
        session = EXTERNAL_SESSIONS.get(url, {})
        added_to_cart = session.get('added_to_cart', False)
        page_loads = session.get('page_loads', 1)

        if is_revisit:
            old_pred = 0
            for h in CAPTURED_HISTORY:
                if h.get('url') == url:
                    old_pred = h['prediction']
                    break
            print(f"[TRACKER] Revisit #{page_loads}: {product_name} | Prediction: {chance}% (was {old_pred}%)")
        else:
            print(f"[TRACKER] New capture: {product_name} | Price: {price} | Category: {category}")
            print(f"[TRACKER] Purchase Prediction: {chance}%")

        # Save to CSV (permanent storage)
        save_to_csv(data, chance, category=category, added_to_cart=added_to_cart, page_loads=page_loads)

        # Add to in-memory history (deduplicated by normalized URL)
        existing_idx = None
        for i, entry in enumerate(CAPTURED_HISTORY):
            if entry.get('url') == url:
                existing_idx = i
                break

        new_entry = {
            "product_name": product_name,
            "price": price,
            "url": url,
            "prediction": chance,
            "timestamp": _time.time(),
            "category": category,
            "category_icon": category_icon,
            "added_to_cart": added_to_cart,
            "page_loads": page_loads,
        }

        if existing_idx is not None:
            # UPDATE existing entry — don't duplicate!
            CAPTURED_HISTORY[existing_idx] = new_entry
            # Move to top of list (most recent)
            CAPTURED_HISTORY.insert(0, CAPTURED_HISTORY.pop(existing_idx))
        else:
            CAPTURED_HISTORY.insert(0, new_entry)
            if len(CAPTURED_HISTORY) > 20:
                CAPTURED_HISTORY.pop()

        return jsonify({
            "status": "captured",
            "prediction": chance,
            "product": product_name,
            "category": category,
            "category_icon": category_icon,
            "added_to_cart": added_to_cart,
            "page_loads": page_loads,
            "is_revisit": is_revisit,
        })
    except Exception as e:
        print(f"[TRACKER] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/external/add-to-cart', methods=['POST', 'OPTIONS'])
def add_to_cart():
    """Receive add-to-cart event — boosts purchase prediction significantly."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        raw_url = data.get('url', 'unknown')
        url = normalize_amazon_url(raw_url)
        product_name = data.get('product_name', 'Unknown Product')
        
        # Normalize URL in data
        data['url'] = url

        # Mark as added to cart in session
        if url in EXTERNAL_SESSIONS:
            EXTERNAL_SESSIONS[url]['added_to_cart'] = True
        else:
            EXTERNAL_SESSIONS[url] = {
                'clicks': 1, 'first_seen': _time.time(), 'last_seen': _time.time(),
                'added_to_cart': True, 'page_loads': 1,
                'category': categorize_product(product_name),
            }

        # Re-calculate prediction with cart boost
        data['added_to_cart'] = True
        chance = calculate_purchase_chance(data)
        category = categorize_product(product_name)

        print(f"[TRACKER] ADD TO CART: {product_name} -> Prediction boosted to {chance}%")

        # Update in-memory history
        for entry in CAPTURED_HISTORY:
            if entry.get('url') == url:
                entry['added_to_cart'] = True
                entry['prediction'] = chance
                break

        # Update CSV
        save_to_csv(data, chance, category=category, added_to_cart=True,
                    page_loads=EXTERNAL_SESSIONS[url].get('page_loads', 1))

        return jsonify({
            "status": "cart_captured",
            "prediction": chance,
            "product": product_name,
            "message": f"Cart event boosted prediction to {chance}%"
        })
    except Exception as e:
        print(f"[TRACKER] Cart tracking error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/external/history')
def external_history():
    """Returns the recent history of captured interactions."""
    return jsonify(CAPTURED_HISTORY)


@app.route('/api/status')
def api_status():
    stats = compute_live_analytics()
    meta = get_model_meta()
    return jsonify({
        'model': 'Correlation-Constrained Latent Factor Model (CCLF)',
        'model_version': 'v4.0',
        'latent_factors': K_LATENT,
        'explained_variance': meta['explained_variance'],
        'data_source': 'Live Amazon Browsing',
        'total_captured': stats['total_interactions'],
        'unique_products': stats['unique_products'],
        'avg_prediction': stats['avg_prediction'],
        'csv_file': CAPTURED_CSV,
        'csv_exists': os.path.exists(CAPTURED_CSV),
        'features': FEATURE_NAMES,
    })


if __name__ == '__main__':
    import sys, io
    # Fix Windows console encoding for emoji
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + "="*60)
    print("  AdNeural - Correlation-Constrained Latent Factor Model")
    print(f"  Model: CCLF with k={K_LATENT} latent factors")
    print("  Waiting for Amazon Safe Tracker extension data...")
    print("  Saving interactions to: " + CAPTURED_CSV)
    print("="*60 + "\n")
    app.run(debug=True)
