import json
import random

# Curated, confirmed working Unsplash IDs for high-end aesthetics
IMAGE_MAPPINGS = {
    "Tech": [
        "1511707171634-5f897ff02aa9", # phone
        "1496181133206-80ce9b88a853", # laptop
        "1505740420928-5e560c06d30e", # headphones
        "1546868871-af0de0ae72be", # tech desk
        "1598327105666-5b89351aff97", # pixel styling
        "1516035069371-29a1b244cc32"  # camera lens
    ],
    "Fashion": [
        "1445205170230-053b83016050", # fashion winter
        "1515886657613-9f3515b0c78f", # shoes
        "1520975954732-86dd53665eb0", # streetwear
        "1489987707025-afc232f7ea0f", # coat
        "1539185441755-769473a23570", # minimalist sneaker
        "1543163521-1bf539c55dd2"  # loafers
    ],
    "Home & Living": [
        "1618220179428-22790b46a015", # aesthetic living room
        "1484101403630-f88118da4aee", # cozy interior
        "1558317374-067fb5f30001", # plant room
        "1585515320310-259814833e62", # modern kitchen
        "1608043152269-423dbba4e7e1"  # smart speaker
    ],
    "Gaming": [
        "1606144042614-b2417e99c4e3", # PS5
        "1593118247619-e2d6f056869e", # Controller neon
        "1550745165-9bc0b252726f", # Neon gaming setup
        "1612287230202-1ff1d85d1bdf", # Stream deck
        "1598550476439-6847785fcea6" # gaming chair
    ],
    "Beauty": [
        "1596462502278-27bfdc403348", # palette
        "1556228578-0d85b1a4d571", # cream
        "1612817288484-6f916006741a", # aesthetic cosmetics
        "1522337360788-8b13dee7a37e", # hair care
        "1620916566398-39f1143ab7be"  # luxurious bottle
    ],
    "Sports & Fitness": [
        "1517836357463-d25dfeac3438", # gym weights
        "1506629082955-511b1aa562c8", # yoga aesthetic
        "1576678927484-cc907957088c", # massage gun
        "1460353581641-37baddab0fa2", # running shoes
        "1602143407151-7111542de6e8"  # hydration bottle
    ],
    "Books & Learning": [
        "1521587760476-6c12a4b040da", # kindle / e-reader
        "1544947950-fa07a98d237f", # aesthetic notebook
        "1495446815901-a7297e633e8d", # library stack
        "1478737270239-2f02b77fc618", # headphones book
        "1516321318423-f06f85e504b3"  # laptop studying
    ],
    "Food & Grocery": [
        "1466637574441-749b8f19452f", # meal kit aesthetic
        "1509042239860-f550ce710b93", # coffee beans
        "1498837167922-ddd27525d352", # fresh produce
        "1622597467836-f3285f2131b8", # green juice
        "1559056199-641a0ac8b55e"  # coffee pour
    ],
    "Travel": [
        "1565026057447-bc90a3dceb87", # luggage airport
        "1553062407-98eeb64c6a62", # travel backpack
        "1473625247510-8ceb1760943f", # tech adapter travel
        "1507842217343-583bb7270b66", # reading on plane
        "1546435770-a3e426bf472b"  # noise cancelling vibe
    ]
}

def fix_images():
    with open('products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
        
    for p in products:
        cat = p['category']
        # Assign a random beautiful image from the category list
        if cat in IMAGE_MAPPINGS:
            img_id = random.choice(IMAGE_MAPPINGS[cat])
            # Use specific w/h and fit parameters for consistency
            p['image'] = f"https://images.unsplash.com/photo-{img_id}?w=600&h=600&fit=crop&q=80"
        else:
            p['image'] = f"https://picsum.photos/seed/{p['id']}/600/600"
            
    with open('products.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)

if __name__ == '__main__':
    fix_images()
    print("products.json images updated with robust Unsplash mapping.")
