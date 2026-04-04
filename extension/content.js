// Amazon Safe Tracker v4 - content.js
// Enhanced: Always tracks page loads (for view count), detects Add to Cart events

console.log("[Amazon-Tracker] ✅ CCLF Tracker Loaded on:", window.location.hostname);

// Track how many times we've sent for THIS page load (prevent re-sends within same load)
let hasSentThisPageLoad = false;
let hasDetectedCart = false;
let currentProduct = { name: "Unknown", price: "N/A", url: "" };

function captureProduct() {
    if (!window.location.href.includes('/dp/')) return;
    if (hasSentThisPageLoad) return; // Only prevent within the SAME page load, not across navigations

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
        hasSentThisPageLoad = true;
        currentProduct = { name: productName, price: price, url: window.location.href };

        // Send to background service worker
        chrome.runtime.sendMessage({
            type: "TRACK_PRODUCT",
            data: {
                product_name: productName,
                price: price,
                url: window.location.href,
                timestamp: new Date().toISOString()
            }
        });

        // Set up Add to Cart detection AFTER product is captured
        setupCartDetection();
    } else {
        console.log("[Amazon-Tracker] ⏳ Still searching for product title...");
    }
}

function setupCartDetection() {
    if (hasDetectedCart) return;
    hasDetectedCart = true;

    // Use event delegation on the document body to catch clicks even if buttons load late
    document.body.addEventListener('click', (e) => {
        const target = e.target;
        
        // Amazon uses various selectors for cart/buy buttons depending on category/layout
        const isCartBtn = target.closest('#add-to-cart-button') || 
                          target.closest('input[name="submit.add-to-cart"]') || 
                          target.closest('[id^="add-to-cart-button"]') ||
                          target.closest('.a-button-input[aria-labelledby*="add-to-cart"]');
                          
        const isBuyNowBtn = target.closest('#buy-now-button') || 
                            target.closest('input[name="submit.buy-now"]');

        if (isCartBtn || isBuyNowBtn) {
            const eventType = isBuyNowBtn ? "buy_now" : "add_to_cart";
            console.log(`[Amazon-Tracker] 🛒 ${eventType.toUpperCase()} detected for:`, currentProduct.name);
            
            chrome.runtime.sendMessage({
                type: "ADD_TO_CART",
                data: {
                    product_name: currentProduct.name,
                    price: currentProduct.price,
                    url: currentProduct.url,
                    event: eventType,
                    timestamp: new Date().toISOString()
                }
            });
        }
    });
    
    console.log("[Amazon-Tracker] 👀 Cart button delegate watcher active");
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
        // Reset for new page — allow tracking again
        hasSentThisPageLoad = false;
        hasDetectedCart = false;
        setTimeout(captureProduct, 2000);
    }
}, 2000);
