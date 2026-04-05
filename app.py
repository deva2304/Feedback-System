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

# ═══════════════════════════════════════════════════════════
# MULTI-DIMENSIONAL SEMANTIC ALIGNMENT ENGINE (MDSA)
# ═══════════════════════════════════════════════════════════
#
#  Novel contribution: textual semantic understanding of products
#  using dense sentence embeddings from a pretrained transformer.
#
#  The MDSA layer works alongside the CCLF behavioral engine to
#  create a Hybrid Two-Stream Architecture:
#    Stream 1 (CCLF): HOW the user behaves (clicks, time, cart)
#    Stream 2 (MDSA): WHAT the product means (semantic similarity)
#
#  This solves the Cold Start Problem — new products with zero
#  behavioral history still get meaningful scores via semantic
#  alignment with previously viewed products.
#
# ═══════════════════════════════════════════════════════════

try:
    from sentence_transformers import SentenceTransformer
    print("[MDSA] Loading semantic embedding model (all-MiniLM-L6-v2)...")
    SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    MDSA_AVAILABLE = True
    EMBEDDING_DIM = 384
    print(f"[MDSA] Model loaded — {EMBEDDING_DIM}-dimensional embeddings ready")
except Exception as e:
    print(f"[MDSA] WARNING: Could not load semantic model: {e}")
    print("[MDSA] Falling back to behavioral-only mode (CCLF)")
    SEMANTIC_MODEL = None
    MDSA_AVAILABLE = False
    EMBEDDING_DIM = 0

# Semantic embedding cache — keyed by product name
SEMANTIC_CACHE = {}

# MDSA Hyperparameters
OMEGA_BEHAVIORAL = 0.60       # Weight for behavioral SVD score
OMEGA_SEMANTIC = 0.40         # Weight for semantic alignment score
OMEGA_COLD_BEHAVIORAL = 0.40  # Cold start: shift weight to semantic
OMEGA_COLD_SEMANTIC = 0.60    # Cold start: semantic dominates
SEMANTIC_DECAY_HOURS = 12     # Time decay half-life for semantic weighting


def get_semantic_embedding(product_name):
    """
    Generate a 384-dimensional dense semantic embedding for a product name.
    
    Uses the all-MiniLM-L6-v2 transformer model to encode the product name
    into a vector that captures its semantic meaning. Results are cached.
    
    Returns: numpy array of shape (384,) or None if MDSA is unavailable
    """
    if not MDSA_AVAILABLE or not product_name:
        return None
    
    # Check cache (normalize key for consistency)
    cache_key = product_name.strip().lower()
    if cache_key in SEMANTIC_CACHE:
        return SEMANTIC_CACHE[cache_key]
    
    try:
        embedding = SEMANTIC_MODEL.encode(product_name, normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)
        SEMANTIC_CACHE[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"[MDSA] Embedding error for '{product_name[:30]}': {e}")
        return None


def compute_semantic_alignment(current_product_name, current_url, all_sessions, history):
    r"""
    Multi-Dimensional Semantic Alignment (MDSA)
    
    Computes how semantically aligned the current product is with the user's
    entire browsing history using dense transformer embeddings.
    
    Mathematical formulation:
      e_i = Encoder(product_name_i)   ∈ R^{384}      (sentence embedding)
      w_j = e^{-Δt_j / \lambda_{sem}}                 (time-decay weight)
      
      sim(e_i, e_j) = (e_i · e_j) / (||e_i|| × ||e_j||)  (cosine similarity)
      
      S_{align} = Σ_j (w_j × sim(e_i, e_j)) / Σ_j w_j    (weighted mean alignment)
    
    Additionally computes 3 semantic dimension scores:
      D1: Functional alignment  — how similar in purpose
      D2: Brand/Quality tier    — premium vs budget clustering  
      D3: Category coherence    — same product family
    
    Returns: dict with semantic_score, dimension_scores, top_similar
    """
    result = {
        'semantic_score': 0.0,
        'top_similar': [],
        'dimension_scores': {
            'functional': 0.0,
            'brand_quality': 0.0,
            'category': 0.0,
        },
        'n_compared': 0,
    }
    
    if not MDSA_AVAILABLE:
        return result
    
    current_embedding = get_semantic_embedding(current_product_name)
    if current_embedding is None:
        return result
    
    now = _time.time()
    similarities = []
    
    # Compare against all products in browsing history
    for entry in history:
        entry_url = entry.get('url', '')
        entry_name = entry.get('product_name', '')
        
        # Skip self-comparison
        if entry_url == current_url or not entry_name:
            continue
        
        other_embedding = get_semantic_embedding(entry_name)
        if other_embedding is None:
            continue
        
        # Cosine similarity (embeddings are already normalized)
        cosine_sim = float(np.dot(current_embedding, other_embedding))
        
        # Time-decay weighting (recent products matter more)
        entry_time = entry.get('timestamp', now)
        hours_ago = (now - entry_time) / 3600
        time_weight = math.exp(-hours_ago / SEMANTIC_DECAY_HOURS)
        
        # Session engagement weight (products with more views contribute more)
        session = all_sessions.get(entry_url, {})
        engagement = math.log(1 + session.get('page_loads', 1))
        
        weighted_sim = cosine_sim * time_weight * engagement
        
        similarities.append({
            'product_name': entry_name,
            'cosine_sim': cosine_sim,
            'weighted_sim': weighted_sim,
            'time_weight': time_weight,
        })
    
    if not similarities:
        return result
    
    # Compute weighted average semantic alignment
    total_weight = sum(s['weighted_sim'] for s in similarities)
    total_raw_weight = sum(s['time_weight'] for s in similarities)
    
    if total_raw_weight > 0:
        weighted_avg = sum(s['cosine_sim'] * s['time_weight'] for s in similarities) / total_raw_weight
    else:
        weighted_avg = 0.0
    
    # FIX 6: Nonlinear semantic calibration
    # Raw cosine sims average 0.1-0.3 for same-domain products.
    # Square-root stretching maps this to 0.31-0.55, giving the semantic
    # stream a usable dynamic range for fusion with behavioral scores.
    calibrated = math.pow(max(weighted_avg, 0.0), 0.5)  # sqrt stretching
    semantic_score = max(0.0, min(calibrated * 100, 100.0))
    
    # Top 3 most semantically similar products
    similarities.sort(key=lambda x: x['cosine_sim'], reverse=True)
    top_similar = [
        {'name': s['product_name'][:60], 'similarity': round(s['cosine_sim'] * 100, 1)}
        for s in similarities[:3]
    ]
    
    # ── Multi-Dimensional Scores ──
    # Dimension 1: Functional alignment (overall semantic similarity)
    functional_score = semantic_score
    
    # Dimension 2: Brand/Quality tier alignment
    # Products in same price tier tend to cluster semantically
    brand_keywords_current = set(current_product_name.lower().split())
    brand_overlap_scores = []
    for s in similarities:
        other_words = set(s['product_name'].lower().split())
        overlap = len(brand_keywords_current & other_words)
        total = max(len(brand_keywords_current | other_words), 1)
        brand_overlap_scores.append(overlap / total * s['time_weight'])
    brand_quality = sum(brand_overlap_scores) / max(sum(s['time_weight'] for s in similarities), 1e-8) * 100
    
    # Dimension 3: Category coherence
    current_category = categorize_product(current_product_name)
    same_cat_count = sum(1 for s in similarities 
                         if categorize_product(s['product_name']) == current_category)
    category_score = (same_cat_count / max(len(similarities), 1)) * 100
    
    result['semantic_score'] = round(semantic_score, 1)
    result['top_similar'] = top_similar
    result['dimension_scores'] = {
        'functional': round(functional_score, 1),
        'brand_quality': round(min(brand_quality, 100), 1),
        'category': round(category_score, 1),
    }
    result['n_compared'] = len(similarities)
    
    return result


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
    'Wearables': '⌚', 'Storage': '💾', 'Keyboards & Mouse': '⌨️',
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
ALPHA_CONFIDENCE = 10   # Confidence scaling: c = 1 + α·log(1+views)
LAMBDA_REG = 0.08       # Tikhonov regularization strength (reduced for sharper SVD)
SIGMOID_SCALE = 2.8     # Sigmoid steepness
SIGMOID_SHIFT = -1.2    # Sigmoid horizontal shift
CART_MULTIPLIER = 1.65  # How much add-to-cart boosts the score
VIEW_LOG_BASE = 1.8     # Logarithmic view scaling base
PLATT_TAU_BASE = 1.0    # Adaptive Platt temperature base

def extract_features(session, data, all_sessions):
    r"""
    Extract an 11-dimensional behavioral feature vector for a product interaction.
    
    Main effects (f1-f8):
      f1: view_count       - Number of times product page was loaded (log-scaled)
      f2: session_duration  - Total time browsing this product (normalized)
      f3: price_signal      - Inverse price signal (cheaper items score higher)
      f4: recency           - How recently the product was last viewed (exponential decay)
      f5: view_velocity     - Views per minute (engagement intensity)
      f6: cart_signal       - Binary: 1 if added to cart, 0 otherwise
      f7: category_affinity - How many products in the same category were viewed
      f8: relative_interest - This product's views relative to average views across all products
    
    Interaction features (f9-f11) — FIX 7:
      f9:  cart_x_views         - Confirmed interest intensity (cart × view_count)
      f10: recency_x_velocity   - Active engagement recency (recency × view_velocity)
      f11: price_x_cart         - Price-conscious cart intent (price_signal × cart_signal)
    
    These interaction terms capture compound behavioral patterns that
    individual features cannot represent, reducing the burden on SVD
    to discover cross-feature correlations from limited data.
    """
    now = _time.time()
    page_loads = session.get('page_loads', 1)
    first_seen = session.get('first_seen', now)
    last_seen = session.get('last_seen', now)
    duration = max(last_seen - first_seen, 1)
    
    # f1: View count (log-scaled to avoid dominance)
    view_count = math.log(1 + page_loads) / math.log(1 + 20)  # Normalize to ~[0,1] (20 views = 1.0)
    
    # f2: Session duration (sigmoid-normalized)
    session_duration = 1 - math.exp(-duration / 300)  # 5 minutes -> ~0.63
    
    # f3: Price signal (inverse — cheaper items get higher signal)
    price_str = str(data.get('price', '0')).replace(',', '').replace('\u20b9', '').replace('$', '').replace('.', '').strip()
    try:
        price_val = float(price_str) if price_str else 0
    except ValueError:
        price_val = 0
    if price_val > 0:
        price_signal = math.exp(-price_val / 100000)  # 1,00,000 -> ~0.37
    else:
        price_signal = 0.5
    
    # f4: Recency (exponential decay — recent views score higher)
    time_since_last = now - last_seen
    recency = math.exp(-time_since_last / 3600)  # 1 hour -> ~0.37
    
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
    category_affinity = min(same_cat_count / 5, 1.0)  # 5+ same-category -> maxed out
    
    # f8: Relative interest (this product's views vs avg views)
    all_loads = [s.get('page_loads', 1) for s in all_sessions.values()]
    avg_loads = sum(all_loads) / max(len(all_loads), 1)
    if avg_loads > 0:
        relative_interest = min(page_loads / (avg_loads * 2), 1.0)
    else:
        relative_interest = 0.5
    
    # ── Interaction Features (FIX 7) ──
    # f9: Cart × Views — confirmed interest intensity
    # A product added to cart AND viewed repeatedly is a strong buy signal
    cart_x_views = cart_signal * view_count
    
    # f10: Recency × Velocity — active engagement recency
    # Recent AND fast-paced viewing indicates current decision-making
    recency_x_velocity = recency * view_velocity
    
    # f11: Price × Cart — price-conscious cart intent
    # Captures whether the user is cart-adding expensive vs cheap items
    price_x_cart = price_signal * cart_signal
    
    return np.array([
        view_count, session_duration, price_signal, recency,
        view_velocity, cart_signal, category_affinity, relative_interest,
        cart_x_views, recency_x_velocity, price_x_cart
    ])

FEATURE_NAMES = [
    'view_count', 'session_duration', 'price_signal', 'recency',
    'view_velocity', 'cart_signal', 'category_affinity', 'relative_interest',
    'cart_x_views', 'recency_x_velocity', 'price_x_cart'
]


def build_correlation_matrix(feature_vectors, time_weights=None):
    r"""
    Build a Time-Decayed Pearson correlation matrix with Tikhonov Regularization.
    
    FIX 1: Proper weighted Pearson normalization.
    
    When time-decay weights W are applied, the denominator must reflect
    the effective sample size from the weight matrix, not the raw count n.
    
    Mathematical formulation:
      X_c = X - \bar{X}_w           (weighted centering)
      C_{decay} = \frac{X_c^T W X_c}{\sum w_i - 1} + \lambda I
    
    Returns: (n_features x n_features) correlation matrix
    """
    n_feat = feature_vectors[0].shape[0] if feature_vectors else 11
    
    if len(feature_vectors) < 2:
        # Not enough data for meaningful correlation — use identity
        return np.eye(n_feat)
    
    # Stack into matrix (n_products x n_features)
    X = np.array(feature_vectors)
    
    # Weighted mean for centering (FIX 1)
    if time_weights is not None:
        w = np.array(time_weights)
        w_sum = np.sum(w)
        if w_sum < 1e-8:
            w = np.ones(X.shape[0])
            w_sum = float(X.shape[0])
        # Weighted mean
        weighted_mean = (w[:, None] * X).sum(axis=0) / w_sum
        X_centered = X - weighted_mean
        W = np.diag(w)
        # Effective sample size for weighted Pearson
        effective_n = w_sum
    else:
        X_centered = X - X.mean(axis=0)
        W = np.eye(X.shape[0])
        effective_n = float(X.shape[0])
    
    # Compute weighted standard deviation for normalization
    if time_weights is not None:
        w_var = (w[:, None] * (X_centered ** 2)).sum(axis=0) / w_sum
        std = np.sqrt(w_var)
    else:
        std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero
    X_normalized = X_centered / std
    
    # Weighted correlation (FIX 1: use effective_n - 1 as denominator)
    denominator = max(effective_n - 1.0, 1.0)
    corr_matrix = (X_normalized.T @ W @ X_normalized) / denominator
    
    # Tikhonov Regularization (L2) to guarantee numerical stability
    corr_matrix += LAMBDA_REG * np.eye(corr_matrix.shape[0])
    
    # Clamp to [-1, 1] for numerical stability (off-diagonal only)
    diag = np.diag(corr_matrix).copy()
    corr_matrix = np.clip(corr_matrix, -1, 1)
    np.fill_diagonal(corr_matrix, diag)  # Preserve diagonal > 1 from regularization
    
    return corr_matrix


def compute_latent_factors(corr_matrix, k=K_LATENT):
    r"""
    Decompose the correlation matrix using SVD to extract latent factors.
    
    R = U \times \Sigma \times V^T
    
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
        
        # Ensure numerical positivity of singular values
        sigma_k = np.maximum(sigma_k, 1e-6)
        
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
    r"""
    Time-Aware Correlation-Constrained Latent Factor Model (TA-CCLF)
    
    Academic Implementation featuring:
    1. Ebbinghaus Time Decay for novelty scaling
    2. Tikhonov (L2) Regularized SVD inversion
    3. Implicit Feedback Confidence Matrix (Hu, Koren & Volinsky, 2008)
    4. Adaptive Temperature-calibrated Platt Scaling probability mapping
    5. L2-normalized user latent vectors with confidence scaling (FIX 2)
    6. Proper CCLF whitening via inverse singular values (FIX 3)
    7. Adaptive score distribution normalization (FIXES 4 & 5)
    
    Mathematical formulation:
      Feature vector:  x_i \in R^{11}
      Time Weights:    W = diag(e^{-t / \lambda_{decay}})
      Correlation:     C_{reg} = X^T W X / (\sum w - 1) + \lambda I   (FIX 1)
      SVD:             C_{reg} = U \Sigma V^T
      Confidence:      c_{uj} = 1 + \alpha \log(1+views) + \beta \cdot cart + \gamma \cdot affinity
      User latent:     p_u = L2Norm(\sum c_{uj} \cdot U^T x_j) \times tanh(C_eff)  (FIX 2)
      Item latent:     q_i = U^T x_i
      Whitened:        s = p_u \cdot q_i    (after \Sigma^{-1} whitening, FIX 3)
      Adaptive Platt:  P(y=1) = \sigma(\alpha_s \cdot \frac{s - \mu_s}{\sigma_s})  (FIX 5)
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
    
    # ── Step 6: Compute User Latent Vector via Implicit Confidence ──
    # (Hu, Koren, Volinsky 2008 — with L2 normalization, FIX 2)
    user_latent = np.zeros(len(sigma_k))
    total_confidence = 0.0
    confidences = []
    
    for i, s_url in enumerate(all_urls):
        s = EXTERNAL_SESSIONS.get(s_url, {})
        
        # c_ui = 1 + alpha*log(views) + beta*cart + gamma*affinity
        c_ui = 1.0 
        c_ui += ALPHA_CONFIDENCE * math.log(1 + s.get('page_loads', 1))
        c_ui += 20.0 if s.get('added_to_cart') else 0.0
        c_ui += 5.0 if s.get('category') == session.get('category') else 0.0
        
        fv = all_feature_vecs[i]
        proj = U_k.T @ fv[:U_k.shape[0]]
        
        user_latent += c_ui * proj
        total_confidence += c_ui
        confidences.append(c_ui)
    
    # FIX 2: L2-normalize then scale by bounded confidence
    # This prevents the user vector from exploding in magnitude while
    # still preserving the confidence-weighted direction.
    user_norm = np.linalg.norm(user_latent)
    if user_norm > 1e-8:
        user_latent = user_latent / user_norm
        # Bounded confidence scaling: tanh maps [0, inf) -> [0, 1)
        # Dividing by (ALPHA_CONFIDENCE * len(all_urls)) normalizes by
        # expected baseline confidence, so only ABOVE-average engagement
        # pushes the magnitude toward 1.0
        expected_baseline = ALPHA_CONFIDENCE * max(len(all_urls), 1) * math.log(2)
        effective_confidence = math.tanh(total_confidence / max(expected_baseline, 1.0))
        user_latent *= effective_confidence
    
    # ── Step 7: Correlation-constrained whitening (FIX 3) ──
    # Proper CCLF whitening uses Sigma^{-1} to decorrelate the latent space.
    # This ensures that correlated features (large sigma) are penalized,
    # and unique discriminative features (small sigma) are amplified.
    # 
    # Without FIX 3, the code used 1/sqrt(sigma) * sigma = sqrt(sigma),
    # which BOOSTED correlated features — the opposite of CCLF design.
    try:
        eigenvalues = np.maximum(sigma_k, 1e-4)
        # Whitening weights: 1/sigma_k, normalized to [0,1]
        whitening_weights = 1.0 / eigenvalues
        whitening_weights /= np.max(whitening_weights)  # Normalize
    except Exception:
        whitening_weights = np.ones(len(sigma_k))
    
    # ── Step 8: Compute whitened dot product score ──
    # score = user_latent . (whitening_weights * item_latent)
    # This is the proper whitened scoring in the latent space
    raw_score = float(np.sum(user_latent * whitening_weights * item_latent))
    
    # FIX 4: NO product-count normalization!
    # With L2-normalized user vectors and bounded confidence scaling,
    # the raw score is naturally bounded in [-1, 1].
    # The old code divided by (1 + n_products * 0.5) which crushed all scores
    # into the flat middle of the sigmoid at ~50%.
    n_products = len(EXTERNAL_SESSIONS)
    
    # ── Step 9: Adaptive Platt Scaling (FIX 5) ──
    # Instead of hardcoded tau/mu, compute the score's position relative
    # to all other products' scores for adaptive calibration.
    all_scores = []
    for i, s_url in enumerate(all_urls):
        fv_i = all_feature_vecs[i]
        item_i = U_k.T @ fv_i[:U_k.shape[0]]
        score_i = float(np.sum(user_latent * whitening_weights * item_i))
        all_scores.append(score_i)
    
    # Z-score normalization against the current product population
    if len(all_scores) >= 2:
        mu_scores = np.mean(all_scores)
        sigma_scores = np.std(all_scores)
        if sigma_scores < 1e-8:
            sigma_scores = 1.0
        z_score = (raw_score - mu_scores) / sigma_scores
    else:
        z_score = raw_score
    
    # Adaptive temperature: tau adapts based on data richness
    tau_adaptive = PLATT_TAU_BASE + 0.5 / max(n_products, 1)
    logit = SIGMOID_SCALE * z_score / tau_adaptive
    
    # Cart signal: strong but bounded intent indicator
    if session.get('added_to_cart'):
        logit += CART_MULTIPLIER
    
    # Cap logit to prevent overflow
    logit_capped = max(min(logit, 10), -10)
    
    probability = 1.0 / (1.0 + math.exp(-logit_capped))
    
    # Scale to percentage
    purchase_pct = probability * 100
    
    # Apply confidence-based range adjustment
    # More data = more confident = wider spread from base rate
    confidence_factor = min(n_products / 4, 1.0)  # Full confidence at 4+ products
    
    # Blend toward base rate for low confidence
    base_rate = 15.0  # Base purchase probability with minimal data
    purchase_pct = base_rate + (purchase_pct - base_rate) * (0.5 + 0.5 * confidence_factor)
    
    # Clamp behavioral score to realistic bounds
    behavioral_pct = max(3.0, min(purchase_pct, 97.0))
    
    # ══════════════════════════════════════════════════════════
    # MULTI-DIMENSIONAL SEMANTIC ALIGNMENT FUSION
    # ══════════════════════════════════════════════════════════
    #
    #  Hybrid Two-Stream Fusion:
    #    final = omega_b * behavioral_score + omega_s * semantic_score
    #
    #  Cold-Start Adaptation:
    #    When page_loads == 1 (first view), semantic weight increases
    #    because there's minimal behavioral data to work with.
    #
    # ══════════════════════════════════════════════════════════
    
    product_name = data.get('product_name', '')
    semantic_result = compute_semantic_alignment(
        product_name, url, EXTERNAL_SESSIONS, CAPTURED_HISTORY
    )
    semantic_pct = semantic_result.get('semantic_score', 0.0)
    
    # Choose fusion weights based on data availability
    page_loads = session.get('page_loads', 1)
    if page_loads <= 1 and MDSA_AVAILABLE:
        # Cold start: trust semantic more
        w_b, w_s = OMEGA_COLD_BEHAVIORAL, OMEGA_COLD_SEMANTIC
    else:
        w_b, w_s = OMEGA_BEHAVIORAL, OMEGA_SEMANTIC
    
    # If MDSA is unavailable or no history exists, use pure behavioral
    if not MDSA_AVAILABLE or semantic_pct == 0.0:
        final_pct = behavioral_pct
    else:
        final_pct = w_b * behavioral_pct + w_s * semantic_pct
    
    # Final clamping
    final_pct = max(3.0, min(final_pct, 97.0))
    
    # Store semantic info for API responses
    session['_last_semantic'] = semantic_result
    session['_last_behavioral_pct'] = round(behavioral_pct, 1)
    
    return round(final_pct, 1)


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
        'type': 'Hybrid TA-CCLF + Multi-Dimensional Semantic Alignment (MDSA)',
        'signals': FEATURE_NAMES,
        'k_latent': K_LATENT,
        'explained_variance': round(explained_var, 1),
        'n_products': len(EXTERNAL_SESSIONS),
        'data_source': 'Live Amazon Browsing (via Browser Extension)',
        'storage': CAPTURED_CSV,
        'total_captured': len(CAPTURED_HISTORY),
        'mdsa_enabled': MDSA_AVAILABLE,
        'embedding_dim': EMBEDDING_DIM,
        'embedding_model': 'all-MiniLM-L6-v2' if MDSA_AVAILABLE else 'N/A',
        'fusion_weights': f'{OMEGA_BEHAVIORAL:.0%} behavioral / {OMEGA_SEMANTIC:.0%} semantic',
        'semantic_cache_size': len(SEMANTIC_CACHE),
    }


# ═══════════════════════════════════════════════════════════
# ANALYTICS — Computed from YOUR real captured data
# ═══════════════════════════════════════════════════════════

# Known brand patterns for extraction
KNOWN_BRANDS = [
    'Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Redmi', 'Poco', 'Realme', 'Oppo', 'Vivo',
    'Sony', 'Bose', 'JBL', 'Boat', 'Noise', 'Nothing', 'Google', 'Motorola', 'Nokia',
    'Lenovo', 'HP', 'Dell', 'Asus', 'Acer', 'MSI', 'LG', 'Philips', 'Panasonic',
    'Nike', 'Adidas', 'Puma', 'Titan', 'Fastrack', 'Fossil', 'Fire-Boltt', 'boAt',
    'CEDO', 'CMFF', 'iQOO', 'Infinix', 'Tecno', 'Honor', 'HTC', 'Marshall', 'Sennheiser',
    'Skullcandy', 'AKG', 'Beats', 'Audio-Technica', 'Jabra', 'Anker', 'Mivi',
]

def extract_brand(product_name):
    """Extract brand name from a product name using known brands + heuristics."""
    if not product_name:
        return 'Unknown'
    
    name_lower = product_name.lower()
    
    # Check known brands first (case-insensitive)
    for brand in KNOWN_BRANDS:
        if brand.lower() in name_lower:
            return brand
    
    # Heuristic: first word is often the brand
    first_word = product_name.split()[0] if product_name.split() else 'Unknown'
    # Only use if it looks like a proper noun (capitalized, 2+ chars)
    if len(first_word) >= 2 and first_word[0].isupper() and first_word.isalpha():
        return first_word
    
    return 'Other'


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
            'top_brands': [],
            'most_viewed': [],
            'top_recommended': [],
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
        'top_brands': compute_top_brands(),
        'most_viewed': compute_most_viewed(),
        'top_recommended': compute_top_recommended(),
    }


def compute_top_brands():
    """Rank brands by total views and number of distinct products browsed."""
    brand_stats = {}
    for h in CAPTURED_HISTORY:
        brand = extract_brand(h.get('product_name', ''))
        if brand not in brand_stats:
            brand_stats[brand] = {'views': 0, 'products': set(), 'cart': 0, 'total_pred': 0}
        brand_stats[brand]['views'] += h.get('page_loads', 1)
        brand_stats[brand]['products'].add(h.get('product_name', ''))
        brand_stats[brand]['total_pred'] += h.get('prediction', 0)
        if h.get('added_to_cart'):
            brand_stats[brand]['cart'] += 1
    
    # Score = views * 2 + distinct_products * 3 + cart_adds * 10
    brand_list = []
    for brand, stats in brand_stats.items():
        score = stats['views'] * 2 + len(stats['products']) * 3 + stats['cart'] * 10
        avg_pred = stats['total_pred'] / max(len(stats['products']), 1)
        brand_list.append({
            'name': brand,
            'views': stats['views'],
            'products_count': len(stats['products']),
            'cart_adds': stats['cart'],
            'score': score,
            'avg_prediction': round(avg_pred, 1),
        })
    
    brand_list.sort(key=lambda x: x['score'], reverse=True)
    return brand_list[:8]


def compute_most_viewed():
    """Rank products by view count (page_loads)."""
    products = []
    for h in CAPTURED_HISTORY:
        products.append({
            'name': h.get('product_name', '')[:60],
            'views': h.get('page_loads', 1),
            'prediction': h.get('prediction', 0),
            'category': h.get('category', 'Other'),
            'brand': extract_brand(h.get('product_name', '')),
            'added_to_cart': h.get('added_to_cart', False),
        })
    products.sort(key=lambda x: x['views'], reverse=True)
    return products[:8]


def compute_top_recommended():
    """
    Rank products by purchase likelihood.
    
    Score combines:
    - Hybrid prediction (behavioral + semantic)
    - View intensity bonus (more views = more interested)
    - Cart bonus
    
    This gives the user's "most likely to purchase" ranking.
    """
    products = []
    for h in CAPTURED_HISTORY:
        pred = h.get('prediction', 0)
        views = h.get('page_loads', 1)
        cart = h.get('added_to_cart', False)
        
        # Composite recommendation score
        rec_score = pred * 0.6 + min(views * 5, 30) + (20 if cart else 0)
        
        products.append({
            'name': h.get('product_name', '')[:60],
            'prediction': pred,
            'views': views,
            'category': h.get('category', 'Other'),
            'brand': extract_brand(h.get('product_name', '')),
            'added_to_cart': cart,
            'rec_score': round(rec_score, 1),
            'semantic_score': h.get('semantic_score', 0),
        })
    products.sort(key=lambda x: x['rec_score'], reverse=True)
    return products[:8]


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def home():
    stats = compute_live_analytics()
    return render_template('index.html',
                           stats=stats,
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

        # Get semantic info from session
        semantic_info = session.get('_last_semantic', {})
        behavioral_score = session.get('_last_behavioral_pct', chance)
        
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
            "semantic_score": semantic_info.get('semantic_score', 0),
            "behavioral_score": behavioral_score,
            "top_similar": semantic_info.get('top_similar', []),
            "dimension_scores": semantic_info.get('dimension_scores', {}),
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
            "semantic_score": semantic_info.get('semantic_score', 0),
            "behavioral_score": behavioral_score,
            "top_similar": semantic_info.get('top_similar', []),
            "dimension_scores": semantic_info.get('dimension_scores', {}),
            "mdsa_enabled": MDSA_AVAILABLE,
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
        'model': 'Hybrid CCLF + MDSA (Multi-Dimensional Semantic Alignment)',
        'model_version': 'v5.0',
        'latent_factors': K_LATENT,
        'explained_variance': meta['explained_variance'],
        'data_source': 'Live Amazon Browsing',
        'total_captured': stats['total_interactions'],
        'unique_products': stats['unique_products'],
        'avg_prediction': stats['avg_prediction'],
        'csv_file': CAPTURED_CSV,
        'csv_exists': os.path.exists(CAPTURED_CSV),
        'features': FEATURE_NAMES,
        'mdsa_enabled': MDSA_AVAILABLE,
        'embedding_model': 'all-MiniLM-L6-v2' if MDSA_AVAILABLE else 'N/A',
        'embedding_dim': EMBEDDING_DIM,
        'fusion_weights': {'behavioral': OMEGA_BEHAVIORAL, 'semantic': OMEGA_SEMANTIC},
        'semantic_cache_size': len(SEMANTIC_CACHE),
    })


if __name__ == '__main__':
    import sys, io
    # Fix Windows console encoding for emoji
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + "="*60)
    print("  AdNeural — Hybrid Two-Stream Purchase Predictor")
    print(f"  Stream 1: CCLF Behavioral Engine (k={K_LATENT} latent factors)")
    print(f"  Stream 2: MDSA Semantic Engine ({'ACTIVE — ' + str(EMBEDDING_DIM) + 'D embeddings' if MDSA_AVAILABLE else 'INACTIVE'})")
    print(f"  Fusion: {OMEGA_BEHAVIORAL:.0%} behavioral + {OMEGA_SEMANTIC:.0%} semantic")
    print("  Waiting for Amazon Safe Tracker extension data...")
    print("  Saving interactions to: " + CAPTURED_CSV)
    print("="*60 + "\n")
    app.run(debug=True)
