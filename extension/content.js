// Amazon Safe Tracker v3 - content.js
// Optimized for reliable product detection and communication via Service Worker

console.log("[Amazon-Tracker] ✅ Reality Tracking Loaded on:", window.location.hostname);

let lastTrackedURL = "";

function captureProduct() {
    if (!window.location.href.includes('/dp/')) return;
    if (window.location.href === lastTrackedURL) return;

    // Direct selectors + aria-label for better coverage
    const titleEl = 
        document.querySelector('#productTitle') || 
        document.querySelector('#title') ||
        document.querySelector('h1.a-size-large span');

    const priceEl = 
        document.querySelector('.a-price-whole') || 
        document.querySelector('.a-offscreen') ||
        document.querySelector('#priceblock_ourprice');

    if (titleEl) {
        const productName = titleEl.innerText.trim();
        const price = priceEl ? priceEl.innerText.trim() : "N/A";
        
        console.log("[Amazon-Tracker] 🎯 Product Found:", productName);
        lastTrackedURL = window.location.href;

        // Send to background service worker (safer than direct fetch)
        chrome.runtime.sendMessage({
            type: "TRACK_PRODUCT",
            data: {
                product_name: productName,
                price: price,
                url: window.location.href,
                timestamp: new Date().toISOString()
            }
        });
    } else {
        console.log("[Amazon-Tracker] ⏳ Still searching for product title...");
    }
}

// Retries ensure we catch the late-loading DOM on Amazon
setTimeout(captureProduct, 1000);
setTimeout(captureProduct, 3000);
setTimeout(captureProduct, 5000);

// Detect in-page navigation (Amazon SPA behavior)
let lastURL = window.location.href;
setInterval(() => {
    if (window.location.href !== lastURL) {
        lastURL = window.location.href;
        lastTrackedURL = ""; 
        setTimeout(captureProduct, 2000);
    }
}, 2000);
