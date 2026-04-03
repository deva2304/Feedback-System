// Amazon Safe Tracker - background.js
// Handles network requests to bypass HTTPS/HTTP mixed content restrictions

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "TRACK_PRODUCT") {
        console.log("[BG-Tracker] 📡 Forwarding to Flask:", message.data.product_name);
        
        fetch('http://127.0.0.1:5000/api/external/track', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message.data)
        })
        .then(response => response.json())
        .then(result => {
            console.log("[BG-Tracker] ✅ Prediction result:", result.prediction);
            // Store for popup display
            chrome.storage.local.set({ 
                "lastPrediction": result.prediction, 
                "lastProduct": message.data.product_name 
            });
        })
        .catch(err => {
            console.error("[BG-Tracker] ❌ Flask Error:", err.message);
        });

        // Mandatory for async sendResponse if we needed one
        return true; 
    }
});
