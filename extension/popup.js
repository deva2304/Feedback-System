// Amazon Safe Tracker - popup.js
document.addEventListener('DOMContentLoaded', () => {
    // Check if we have any captured prediction data in chrome.storage
    chrome.storage.local.get(["lastPrediction", "lastProduct"], (data) => {
        if (data.lastProduct) {
            document.getElementById('product').innerText = data.lastProduct;
            
            if (data.lastPrediction !== undefined) {
                const score = data.lastPrediction;
                const scoreEl = document.getElementById('score');
                scoreEl.innerText = score + "%";
                
                // Color coding the score
                if (score > 70) scoreEl.style.color = "#4CAF50"; // Green
                else if (score > 40) scoreEl.style.color = "#ff9900"; // Orange
                else scoreEl.style.color = "#f44336"; // Red
            }
        }
    });
});
