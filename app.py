from flask import Flask, render_template, request, jsonify
import os
import json
import csv
import time as _time
from datetime import datetime
import random

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

# Load any previously saved data on startup
def load_captured_history():
    """Load previously captured interactions from CSV on server start."""
    if not os.path.exists(CAPTURED_CSV):
        return
    try:
        with open(CAPTURED_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                CAPTURED_HISTORY.append({
                    'product_name': row.get('product_name', 'Unknown'),
                    'price': row.get('price', 'N/A'),
                    'url': row.get('url', ''),
                    'prediction': float(row.get('prediction', 0)),
                    'timestamp': float(row.get('timestamp', 0)),
                })
        # Keep only last 20
        while len(CAPTURED_HISTORY) > 20:
            CAPTURED_HISTORY.pop(0)
        print(f"[TRACKER] Loaded {len(CAPTURED_HISTORY)} previous interactions from {CAPTURED_CSV}")
    except Exception as e:
        print(f"[TRACKER] Could not load history: {e}")


def save_to_csv(data, prediction):
    """Append a captured interaction to the CSV file for permanent storage."""
    file_exists = os.path.exists(CAPTURED_CSV)
    try:
        with open(CAPTURED_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'product_name', 'price', 'url', 'prediction', 'captured_at'])
            writer.writerow([
                _time.time(),
                data.get('product_name', 'Unknown'),
                data.get('price', 'N/A'),
                data.get('url', ''),
                prediction,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
    except Exception as e:
        print(f"[TRACKER] CSV save error: {e}")


# Load history on startup
load_captured_history()


# ═══════════════════════════════════════════════════════════
# REAL-TIME PURCHASE PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════

def calculate_purchase_chance(data):
    """
    Real-time Purchase Probability Prediction.

    Uses behavioral signals captured from the user's browsing:
      1. Click frequency — how many times they visited the product
      2. Session duration — how long they've been browsing
      3. Price sensitivity — cheaper items get a slight boost
      4. Repeated interest — coming back to same product = strong signal
    """
    url = data.get('url', 'unknown')
    now = _time.time()

    # Initialize or update session
    if url not in EXTERNAL_SESSIONS:
        EXTERNAL_SESSIONS[url] = {'clicks': 0, 'first_seen': now, 'last_seen': now}

    session = EXTERNAL_SESSIONS[url]
    session['clicks'] += 1
    session['last_seen'] = now

    # ── Factor 1: Click frequency (max 40 points) ──
    click_score = min(session['clicks'] * 12, 40)

    # ── Factor 2: Time spent browsing (max 25 points) ──
    duration = now - session['first_seen']
    time_score = min(duration * 0.4, 25)

    # ── Factor 3: Price signal (max 15 points) ──
    price_str = str(data.get('price', '0')).replace(',', '').replace('₹', '').replace('$', '').strip()
    try:
        price_val = float(price_str)
    except ValueError:
        price_val = 0
    # Lower price → slightly higher chance
    if price_val > 0:
        price_score = max(0, 15 - (price_val / 5000) * 10)
    else:
        price_score = 8

    # ── Factor 4: Return visits (max 15 points) ──
    return_gap = now - session.get('last_seen', now)
    return_score = min(session['clicks'] * 5, 15) if session['clicks'] > 1 else 0

    # ── Factor 5: Small random noise for realism ──
    noise = random.uniform(1, 5)

    total = click_score + time_score + price_score + return_score + noise
    return round(min(total, 98.5), 1)


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
        }

    predictions = [h['prediction'] for h in CAPTURED_HISTORY]
    unique_names = set(h['product_name'] for h in CAPTURED_HISTORY)

    return {
        'total_products_viewed': len(CAPTURED_HISTORY),
        'total_interactions': len(CAPTURED_HISTORY),
        'avg_prediction': round(sum(predictions) / len(predictions), 1),
        'highest_prediction': round(max(predictions), 1),
        'unique_products': len(unique_names),
        'products': CAPTURED_HISTORY[:10],
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
    return render_template('about.html', model_meta={
        'type': 'Real-Time Behavioral Analysis',
        'signals': ['Click Frequency', 'Session Duration', 'Price Sensitivity', 'Return Visits'],
        'data_source': 'Live Amazon Browsing (via Browser Extension)',
        'storage': CAPTURED_CSV,
        'total_captured': len(CAPTURED_HISTORY),
    })


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

        print(f"[TRACKER] ✅ Captured: {product_name} | Price: {price}")

        # Calculate prediction
        chance = calculate_purchase_chance(data)
        print(f"[TRACKER] 🎯 Purchase Prediction: {chance}%")

        # Save to CSV (permanent storage)
        save_to_csv(data, chance)

        # Add to in-memory history (for live dashboard)
        CAPTURED_HISTORY.insert(0, {
            "product_name": product_name,
            "price": price,
            "url": data.get('url', ''),
            "prediction": chance,
            "timestamp": _time.time()
        })
        # Keep last 20
        if len(CAPTURED_HISTORY) > 20:
            CAPTURED_HISTORY.pop()

        return jsonify({
            "status": "captured",
            "prediction": chance,
            "product": product_name
        })
    except Exception as e:
        print(f"[TRACKER] ❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/external/history')
def external_history():
    """Returns the recent history of captured interactions."""
    return jsonify(CAPTURED_HISTORY)


@app.route('/api/status')
def api_status():
    stats = compute_live_analytics()
    return jsonify({
        'model': 'Real-Time Behavioral Purchase Predictor',
        'data_source': 'Live Amazon Browsing',
        'total_captured': stats['total_interactions'],
        'avg_prediction': stats['avg_prediction'],
        'csv_file': CAPTURED_CSV,
        'csv_exists': os.path.exists(CAPTURED_CSV),
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🧠 AdNeural — Real-Time Purchase Predictor")
    print("  📡 Waiting for Amazon Safe Tracker extension data...")
    print("  💾 Saving interactions to: " + CAPTURED_CSV)
    print("="*60 + "\n")
    app.run(debug=True)
