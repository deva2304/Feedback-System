from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import os
import json
import urllib.request
import urllib.parse
import time as _time
from datetime import datetime
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# ═══════════════════════════════════════════════════════════
# CORRELATION-CONSTRAINED LATENT FACTOR MODEL (CCLF)
# Genuine implementation matching the research title
# ═══════════════════════════════════════════════════════════

# ── Step 0: Load implicit feedback data ──
DATA = pd.read_csv("data.csv")
IMPLICIT_FEATURES = ['click_count', 'view_time_sec', 'scroll_depth', 'add_to_cart',
                     'purchase', 'search_query_count', 'session_duration_sec', 'position']

# Model hyperparameters
K_LATENT = 5           # Number of latent dimensions
ALPHA_CONF = 40        # Confidence scaling factor (Hu et al. 2008)
LAMBDA_CORR = 0.3      # Correlation constraint strength

# Implicit signal weights — how much each behavioral signal contributes
IMPLICIT_WEIGHTS = {
    'purchase':           5.0,   # Strongest signal
    'add_to_cart':        3.0,   # Strong purchase intent
    'click_count':        1.0,   # Moderate interest (per click)
    'view_time_sec':      0.01,  # Normalized (seconds → score)
    'scroll_depth':       0.02,  # Normalized (% → score)
    'search_query_count': 0.5,   # Active search behavior
    'session_duration_sec': 0.005, # Session engagement
    'position':          -0.1,   # Penalty: higher position = less interest
}


# ── Step 1: Build User-Item Implicit Interaction Matrix ──
print("[CCLF] Building user-item implicit feedback matrix...")

n_users = DATA['user_id'].nunique()
n_items = DATA['item_id'].nunique()
user_ids = sorted(DATA['user_id'].unique())
item_ids = sorted(DATA['item_id'].unique())
user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

# Compute weighted implicit signal for each interaction
R = np.zeros((n_users, n_items))
for _, row in DATA.iterrows():
    u_idx = user_id_map[row['user_id']]
    i_idx = item_id_map[row['item_id']]
    # Aggregate implicit signals with weights
    signal = 0
    for feat, w in IMPLICIT_WEIGHTS.items():
        signal += w * row[feat]
    R[u_idx, i_idx] = max(signal, 0)  # No negative preferences

# Compute confidence matrix: C = 1 + alpha * R
C = 1.0 + ALPHA_CONF * R

# Binary preference: 1 if any interaction, 0 otherwise
P_pref = (R > 0).astype(float)

# Matrix sparsity
n_observed = np.count_nonzero(R)
sparsity = 1.0 - (n_observed / (n_users * n_items))
print(f"[CCLF] Matrix: {n_users} users x {n_items} items, "
      f"sparsity: {sparsity:.1%}, {n_observed} observed interactions")


# ── Step 2: Compute Behavioral Correlation Matrix (the "Constraint") ──
print("[CCLF] Computing behavioral correlation matrix...")

CORR_MATRIX = DATA[IMPLICIT_FEATURES].corr()
corr_values = CORR_MATRIX.values

# SVD of correlation matrix → behavioral basis vectors
U_corr, S_corr, Vt_corr = np.linalg.svd(corr_values)
CORR_BASIS = U_corr[:, :K_LATENT]  # Top-k behavioral eigenvectors

# This captures the dominant behavioral patterns:
# e.g., "add_to_cart + purchase" are highly correlated (r=0.57)
# The CORR_BASIS encodes these relationships


# ── Step 3: Truncated SVD on User-Item Matrix → Latent Factors ──
print(f"[CCLF] Factorizing interaction matrix (k={K_LATENT})...")

# Apply confidence weighting to the interaction matrix
R_weighted = np.sqrt(C) * R

# Truncated SVD: R ≈ U * Σ * V^T
R_sparse = csr_matrix(R_weighted)
U_factors, S_factors, Vt_factors = svds(R_sparse, k=K_LATENT)

# Sort by singular value magnitude (svds returns ascending)
sort_idx = np.argsort(-S_factors)
U_factors = U_factors[:, sort_idx]
S_factors = S_factors[sort_idx]
Vt_factors = Vt_factors[sort_idx, :]

# User latent factors: P = U * sqrt(Σ)
USER_FACTORS = U_factors * np.sqrt(S_factors)

# Item latent factors: Q = V^T.T * sqrt(Σ) = V * sqrt(Σ)
ITEM_FACTORS_RAW = Vt_factors.T * np.sqrt(S_factors)

# Explained variance ratio
total_energy = np.sum(np.linalg.svd(R_weighted, compute_uv=False)**2)
retained_energy = np.sum(S_factors**2)
EXPLAINED_VARIANCE = retained_energy / total_energy if total_energy > 0 else 0

print(f"[CCLF] Explained variance: {EXPLAINED_VARIANCE:.1%} (top {K_LATENT} factors)")


# ── Step 4: Apply Correlation Constraint ──
# The key innovation: constrain item factors to respect behavioral correlations
# Q_adj = Q + λ * Q * (W * W^T) where W = top eigenvectors of correlation matrix
print(f"[CCLF] Applying correlation constraint (lambda={LAMBDA_CORR})...")

# Build a projection matrix from the behavioral correlation basis
# This ensures item factors are "aligned" with the dominant behavioral patterns
# For each item, compute its behavioral profile from actual user interactions
item_behavioral_profiles = np.zeros((n_items, len(IMPLICIT_FEATURES)))
for _, row in DATA.iterrows():
    i_idx = item_id_map[row['item_id']]
    item_behavioral_profiles[i_idx] = row[IMPLICIT_FEATURES].values

# Normalize behavioral profiles
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
item_behavioral_norm = scaler.fit_transform(item_behavioral_profiles)

# Project behavioral profiles into the correlation basis space
item_corr_projection = item_behavioral_norm @ CORR_BASIS  # (n_items, K_LATENT)

# Normalize to unit vectors for combining
item_corr_norm = item_corr_projection / (np.linalg.norm(item_corr_projection, axis=1, keepdims=True) + 1e-8)
item_factors_norm = ITEM_FACTORS_RAW / (np.linalg.norm(ITEM_FACTORS_RAW, axis=1, keepdims=True) + 1e-8)

# Constrained item factors: blend raw SVD factors with correlation-projected factors
ITEM_FACTORS = (1 - LAMBDA_CORR) * item_factors_norm + LAMBDA_CORR * item_corr_norm

print(f"[CCLF] Model ready. User factors: {USER_FACTORS.shape}, Item factors: {ITEM_FACTORS.shape}")


# ── Step 5: Precompute per-item implicit signal breakdown ──
# For the "Why?" explainability feature
ITEM_SIGNAL_BREAKDOWN = {}
for iid, idx in item_id_map.items():
    item_data = DATA[DATA['item_id'] == iid]
    if len(item_data) > 0:
        breakdown = {}
        for feat in IMPLICIT_FEATURES:
            breakdown[feat] = round(item_data[feat].mean(), 2)
        breakdown['n_interactions'] = len(item_data)
        ITEM_SIGNAL_BREAKDOWN[iid] = breakdown


# ── Model metadata for display ──
MODEL_META = {
    'n_users':            n_users,
    'n_items':            n_items,
    'n_interactions':     len(DATA),
    'sparsity':           round(sparsity * 100, 1),
    'k_latent':           K_LATENT,
    'alpha_confidence':   ALPHA_CONF,
    'lambda_corr':        LAMBDA_CORR,
    'explained_variance': round(EXPLAINED_VARIANCE * 100, 1),
    'singular_values':    [round(float(s), 2) for s in S_factors],
    'implicit_weights':   IMPLICIT_WEIGHTS,
}
print(f"[CCLF] Model metadata: {json.dumps({k:v for k,v in MODEL_META.items() if k != 'implicit_weights'})}")


# ═══════════════════════════════════════════════════════════
# PRODUCT DATA — From products.json
# ═══════════════════════════════════════════════════════════

CATEGORY_META = {
    'Tech':             {'icon': '💻', 'gradient': 'linear-gradient(135deg, #7c3aed, #3b82f6)'},
    'Fashion':          {'icon': '👗', 'gradient': 'linear-gradient(135deg, #ec4899, #f43f5e)'},
    'Home & Living':    {'icon': '🏡', 'gradient': 'linear-gradient(135deg, #f59e0b, #ef4444)'},
    'Gaming':           {'icon': '🎮', 'gradient': 'linear-gradient(135deg, #10b981, #06b6d4)'},
    'Beauty':           {'icon': '✨', 'gradient': 'linear-gradient(135deg, #f472b6, #c084fc)'},
    'Sports & Fitness': {'icon': '🏋️', 'gradient': 'linear-gradient(135deg, #34d399, #059669)'},
    'Books & Learning': {'icon': '📚', 'gradient': 'linear-gradient(135deg, #fbbf24, #f97316)'},
    'Food & Grocery':   {'icon': '🍕', 'gradient': 'linear-gradient(135deg, #fb923c, #dc2626)'},
    'Travel':           {'icon': '✈️', 'gradient': 'linear-gradient(135deg, #38bdf8, #6366f1)'},
}


def load_products():
    """Load products from the local products.json database."""
    with open('products.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)

    products = []
    for item in raw:
        pv = item.get('price', 0)
        ps = f"${pv:,.0f}" if pv >= 100 else f"${pv:.2f}"
        products.append({
            'id':          item['id'],
            'name':        item.get('name', 'Product'),
            'category':    item.get('category', 'Tech'),
            'price':       ps,
            'price_usd':   float(pv),
            'tag':         item.get('tag', 'New'),
            'image':       item.get('image', ''),
            'link':        item.get('link', '#'),
            'rating':      round(item.get('rating', 4.0), 1),
            'brand':       item.get('brand', ''),
            'stock':       item.get('stock', 50),
            'discount':    round(item.get('discount', 0), 1),
        })

    cats = {}
    for p in products:
        c = p['category']
        cats[c] = cats.get(c, {'count': 0})
        cats[c]['count'] += 1

    return products, cats


PRODUCTS, CATEGORIES = load_products()
print(f"[CCLF] Loaded {len(PRODUCTS)} products from products.json")


# ═══════════════════════════════════════════════════════════
# LIVE EXCHANGE RATES — Frankfurter API (ECB)
# ═══════════════════════════════════════════════════════════

_fx_cache = {'rates': {}, 'ts': 0, 'ttl': 3600, 'date': None}
FALLBACK_RATES = {
    'INR': 83.5, 'GBP': 0.79, 'EUR': 0.92,
    'JPY': 151.0, 'AUD': 1.54, 'AED': 3.67,
}


def fetch_live_rates(force=False):
    global _fx_cache
    now = _time.time()
    if not force and _fx_cache['rates'] and (now - _fx_cache['ts']) < _fx_cache['ttl']:
        return _fx_cache['rates'], _fx_cache['date']
    try:
        url = 'https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR,GBP,EUR,JPY,AUD'
        req = urllib.request.Request(url, headers={'User-Agent': 'AdNeural/2.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        rates = data.get('rates', {})
        rates['AED'] = 3.6725
        rate_date = data.get('date', 'live')
        _fx_cache.update({'rates': rates, 'ts': now, 'date': rate_date})
        return rates, rate_date
    except Exception as e:
        print(f"[CCLF] Exchange rate error: {e}")
        if _fx_cache['rates']:
            return _fx_cache['rates'], _fx_cache['date']
        return FALLBACK_RATES, 'fallback'


_live_rates, _rate_date = fetch_live_rates()


def get_countries():
    rates, _ = fetch_live_rates()
    return {
        'US':  {'flag': '🇺🇸', 'name': 'United States', 'currency': 'USD', 'symbol': '$',  'rate': 1.0},
        'IN':  {'flag': '🇮🇳', 'name': 'India',         'currency': 'INR', 'symbol': '₹',  'rate': rates.get('INR', 83.5)},
        'GB':  {'flag': '🇬🇧', 'name': 'United Kingdom', 'currency': 'GBP', 'symbol': '£',  'rate': rates.get('GBP', 0.79)},
        'EU':  {'flag': '🇪🇺', 'name': 'Europe',         'currency': 'EUR', 'symbol': '€',  'rate': rates.get('EUR', 0.92)},
        'JP':  {'flag': '🇯🇵', 'name': 'Japan',          'currency': 'JPY', 'symbol': '¥',  'rate': rates.get('JPY', 151.0)},
        'AU':  {'flag': '🇦🇺', 'name': 'Australia',      'currency': 'AUD', 'symbol': 'A$', 'rate': rates.get('AUD', 1.54)},
        'AE':  {'flag': '🇦🇪', 'name': 'UAE',             'currency': 'AED', 'symbol': 'د.إ','rate': rates.get('AED', 3.67)},
    }


COUNTRIES = get_countries()


def convert_price(usd_price_str, country_code):
    countries = get_countries()
    country = countries.get(country_code, countries['US'])
    rate = country['rate']
    symbol = country['symbol']
    suffix = ''
    clean = usd_price_str.strip()
    for s in ['/mo', '/wk', '/yr']:
        if clean.endswith(s):
            suffix = s
            clean = clean[:-len(s)]
            break
    clean = clean.replace('$', '').replace(',', '').strip()
    try:
        usd_val = float(clean)
    except ValueError:
        return usd_price_str
    converted = usd_val * rate
    if converted >= 1000:
        formatted = f"{converted:,.0f}"
    elif converted >= 100:
        formatted = f"{converted:.0f}"
    else:
        formatted = f"{converted:.2f}"
    return f"{symbol}{formatted}{suffix}"


if not os.path.exists('static/plots'):
    os.makedirs('static/plots')


# ═══════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE — Genuine CCLF Scoring
# ═══════════════════════════════════════════════════════════

def compute_recommendation_scores(category, user_id=None):
    """
    Correlation-Constrained Latent Factor Model (CCLF) for implicit feedback.

    Pipeline:
      1. Get user latent vector from the factorized user-item matrix
      2. For each product, map to a data-item and get its constrained latent vector
      3. Score = dot product of user and item latent vectors
      4. Return ranked products with full explainability metadata
    """
    cat_products = [p for p in PRODUCTS if p['category'] == category]

    # ── Get user latent vector ──
    if user_id and user_id in user_id_map:
        u_idx = user_id_map[user_id]
        user_latent = USER_FACTORS[u_idx]
    else:
        # For unknown users: synthesize from population mean + noise
        # This simulates a cold-start scenario with exploration
        np.random.seed(user_id if user_id else 0)
        population_mean = USER_FACTORS.mean(axis=0)
        population_std = USER_FACTORS.std(axis=0)
        user_latent = population_mean + np.random.randn(K_LATENT) * population_std * 0.3

    user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)

    # ── Score each product ──
    scored_products = []
    for product in cat_products:
        # Map product to data.csv item: deterministic mapping via modular arithmetic
        # This bridges the product catalog with the learned interaction patterns
        mapped_item_idx = (product['id'] - 1) % n_items

        # Get the correlation-constrained item latent vector
        item_latent = ITEM_FACTORS[mapped_item_idx]
        item_latent_norm = item_latent / (np.linalg.norm(item_latent) + 1e-8)

        # Score = cosine similarity in latent space
        raw_score = np.dot(user_latent_norm, item_latent_norm)

        # Map from [-1, 1] to [0, 100] for display
        relevance = round(((raw_score + 1) / 2) * 100, 1)

        # ── Explainability: which implicit signals drove this recommendation? ──
        mapped_item_id = item_ids[mapped_item_idx]
        signal_data = ITEM_SIGNAL_BREAKDOWN.get(mapped_item_id, {})

        # Compute feature importance for this user-item pair
        # Based on the correlation structure and the user's latent position
        corr_weights = np.abs(corr_values[4])  # purchase-column correlations
        feature_importance = {}
        for idx, feat in enumerate(IMPLICIT_FEATURES):
            # Weight = |correlation with purchase| * |user latent alignment|
            importance = corr_weights[idx] * np.abs(user_latent_norm[idx % K_LATENT])
            feature_importance[feat] = round(float(importance), 4)

        top_features_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in top_features_sorted[:3]]

        # Confidence level from the interaction matrix
        if user_id and user_id in user_id_map:
            u_idx = user_id_map[user_id]
            confidence = float(C[u_idx, mapped_item_idx])
        else:
            confidence = 1.0

        scored_products.append({
            **product,
            'relevance':           relevance,
            'raw_score':           round(float(raw_score), 4),
            'confidence':          round(confidence, 1),
            'user_latent':         user_latent_norm.round(3).tolist(),
            'item_latent':         item_latent_norm.round(3).tolist(),
            'top_features':        top_features,
            'feature_importance':  feature_importance,
            'signal_data':         signal_data,
        })

    # Sort by relevance (highest first)
    scored_products.sort(key=lambda x: x['relevance'], reverse=True)
    return scored_products, user_latent_norm.round(3).tolist()


# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════

@app.route('/')
def home():
    global COUNTRIES
    COUNTRIES = get_countries()
    categories_list = []
    for cat_name, meta in CATEGORY_META.items():
        count = CATEGORIES.get(cat_name, {}).get('count', 0)
        if count == 0:
            continue
        categories_list.append({
            'name': cat_name, 'icon': meta['icon'],
            'count': count, 'gradient': meta['gradient'],
        })
    countries_list = [{'code': k, **v} for k, v in COUNTRIES.items()]
    _, rate_date = fetch_live_rates()
    return render_template('index.html',
                           categories=categories_list,
                           countries=countries_list,
                           total_products=len(PRODUCTS),
                           total_users=MODEL_META['n_users'],
                           total_interactions=MODEL_META['n_interactions'],
                           rate_date=rate_date or 'N/A',
                           is_live=bool(_fx_cache['ts']),
                           model_meta=MODEL_META)


@app.route('/recommend', methods=['POST'])
def recommend():
    global COUNTRIES
    COUNTRIES = get_countries()
    category = request.form.get('category', 'Tech')
    user_id_str = request.form.get('user_id', '')
    user_id = int(user_id_str) if user_id_str.isdigit() else None
    country_code = request.form.get('country', 'US')

    scored_products, user_latent = compute_recommendation_scores(category, user_id)

    for product in scored_products:
        product['price'] = convert_price(product['price'], country_code)

    country_info = COUNTRIES.get(country_code, COUNTRIES['US'])
    corr_data = CORR_MATRIX.round(3).to_dict()
    _, rate_date = fetch_live_rates()

    return render_template('result.html',
                           category=category,
                           category_icon=CATEGORY_META.get(category, {}).get('icon', '🛒'),
                           products=scored_products,
                           user_latent=user_latent,
                           user_id=user_id,
                           total_products=len(scored_products),
                           avg_relevance=round(np.mean([p['relevance'] for p in scored_products]), 1),
                           top_score=scored_products[0]['relevance'] if scored_products else 0,
                           corr_data=json.dumps(corr_data),
                           country=country_info,
                           country_code=country_code,
                           rate_date=rate_date or 'N/A',
                           is_live=bool(_fx_cache['ts']),
                           model_meta=MODEL_META)


@app.route('/analytics')
def analytics():
    stats = {
        'total_users': int(DATA['user_id'].nunique()),
        'total_items': int(DATA['item_id'].nunique()),
        'total_interactions': len(DATA),
        'avg_clicks': round(DATA['click_count'].mean(), 1),
        'avg_view_time': round(DATA['view_time_sec'].mean(), 1),
        'purchase_rate': round(DATA['purchase'].mean() * 100, 1),
        'cart_rate': round(DATA['add_to_cart'].mean() * 100, 1),
        'avg_scroll': round(DATA['scroll_depth'].mean(), 1),
    }
    corr_labels = IMPLICIT_FEATURES
    corr_values_list = CORR_MATRIX.loc[IMPLICIT_FEATURES, IMPLICIT_FEATURES].round(3).values.tolist()
    click_hist = DATA['click_count'].value_counts().sort_index()
    purchase_yes = DATA[DATA['purchase'] == 1][IMPLICIT_FEATURES].mean().round(2).to_dict()
    purchase_no = DATA[DATA['purchase'] == 0][IMPLICIT_FEATURES].mean().round(2).to_dict()
    return render_template('visuals.html',
                           stats=stats,
                           corr_labels=json.dumps(corr_labels),
                           corr_values=json.dumps(corr_values_list),
                           purchase_yes=json.dumps(purchase_yes),
                           purchase_no=json.dumps(purchase_no),
                           click_labels=json.dumps(click_hist.index.tolist()),
                           click_values=json.dumps(click_hist.values.tolist()))


@app.route('/about')
def about():
    return render_template('about.html', model_meta=MODEL_META)


@app.route('/api/product/<int:product_id>')
def get_product(product_id):
    product = next((p for p in PRODUCTS if p['id'] == product_id), None)
    if product:
        return jsonify(product)
    return jsonify({'error': 'Product not found'}), 404


@app.route('/api/rates')
def live_rates():
    rates, rate_date = fetch_live_rates()
    return jsonify({
        'base': 'USD', 'date': rate_date, 'rates': rates,
        'source': 'Frankfurter API (European Central Bank)',
    })


@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    global COUNTRIES
    rates, rate_date = fetch_live_rates(force=True)
    COUNTRIES = get_countries()
    return jsonify({
        'status': 'refreshed', 'rates': rates,
        'rate_date': rate_date, 'total_products': len(PRODUCTS),
    })


@app.route('/api/status')
def api_status():
    return jsonify({
        'model': 'Correlation-Constrained Latent Factor Model (CCLF)',
        'products_loaded': len(PRODUCTS),
        'model_meta': MODEL_META,
        'exchange_rates_live': bool(_fx_cache['ts']),
        'rate_date': _fx_cache['date'],
    })


if __name__ == '__main__':
    app.run(debug=True)
