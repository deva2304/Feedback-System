// Amazon Safe Tracker v4 - background.js
// Handles TRACK_PRODUCT and ADD_TO_CART messages from content.js

const FLASK_SERVER = 'http://127.0.0.1:5000';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "TRACK_PRODUCT") {
        console.log("[BG-Tracker] 📡 Forwarding to Flask:", message.data.product_name);
        
        fetch(`${FLASK_SERVER}/api/external/track`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message.data)
        })
        .then(response => response.json())
        .then(result => {
            const status = result.is_revisit ? '🔄 Revisit' : '✅ New';
            console.log(`[BG-Tracker] ${status} | Prediction: ${result.prediction}% | Views: ${result.page_loads}`);
            
            // Store for popup display
            chrome.storage.local.set({ 
                "lastPrediction": result.prediction, 
                "lastProduct": message.data.product_name,
                "lastPageLoads": result.page_loads,
                "lastIsRevisit": result.is_revisit,
                "lastAddedToCart": result.added_to_cart
            });
        })
        .catch(err => {
            console.error("[BG-Tracker] ❌ Flask Error:", err.message);
        });

        return true;
    }

    if (message.type === "ADD_TO_CART") {
        console.log("[BG-Tracker] 🛒 Cart event for:", message.data.product_name);
        
        fetch(`${FLASK_SERVER}/api/external/add-to-cart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message.data)
        })
        .then(response => response.json())
        .then(result => {
            console.log("[BG-Tracker] 🛒 Cart prediction boosted to:", result.prediction + "%");
            
            // Update stored prediction with cart-boosted value
            chrome.storage.local.set({
                "lastPrediction": result.prediction,
                "lastProduct": message.data.product_name,
                "lastAddedToCart": true
            });
        })
        .catch(err => {
            console.error("[BG-Tracker] ❌ Cart tracking error:", err.message);
        });

        return true;
    }
});
