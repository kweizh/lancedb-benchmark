FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        "voyageai>=0.3,<0.5" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Seed products.json with 60 deterministic English product descriptions
# spread across 4 categories. Four anchor products carry highly distinctive
# wording so the Voyage AI 'voyage-3' embeddings make them the clear top-1
# match for the four verifier anchor queries.
RUN python3 - <<'PYEOF'
import json, os

products = []

# --- Anchor products (one per category, distinctive wording) ---
products.append({
    "id": "prod-electronics-headphones-anc",
    "category": "electronics",
    "description": "Premium over-ear wireless Bluetooth headphones with industry-leading active noise cancellation, 40-hour battery life, and travel-friendly folding design ideal for long flights and daily commutes.",
})
products.append({
    "id": "prod-kitchen-espresso-machine",
    "category": "kitchen",
    "description": "Semi-automatic high-pressure 15-bar espresso machine with built-in conical burr grinder, steam wand for latte art, and stainless steel boiler designed for home baristas crafting cafe-quality espresso shots.",
})
products.append({
    "id": "prod-fitness-carbon-marathon-shoes",
    "category": "fitness",
    "description": "Ultra-lightweight carbon-fiber plated marathon racing shoes engineered for elite road runners, featuring a responsive PEBA midsole foam, breathable mesh upper, and energy-return geometry designed to break personal records on 26.2 mile races.",
})
products.append({
    "id": "prod-books-dragon-wizard-fantasy",
    "category": "books",
    "description": "International bestselling epic fantasy novel chronicling a young apprentice wizard who befriends an ancient dragon and must master arcane magic to save the kingdom; perfect for fans of high-fantasy coming-of-age sagas.",
})

# --- Filler / distractor products: 14 per category x 4 = 56 ---
electronics_extras = [
    ("prod-elec-soundbar-living-room",          "Compact 2.1 channel Dolby Atmos soundbar with wireless subwoofer and HDMI eARC, designed to upgrade movie nights in living rooms with rich cinematic surround sound."),
    ("prod-elec-smart-tv-65",                   "65 inch 4K QLED smart television with HDR10+, 120Hz refresh rate, and integrated streaming platform for binge-watching shows in ultra-high definition."),
    ("prod-elec-mechanical-keyboard",           "Hot-swappable mechanical gaming keyboard with RGB per-key backlighting, tactile brown switches, and aluminum frame engineered for competitive esports players."),
    ("prod-elec-ergonomic-mouse",               "Vertical ergonomic wireless mouse shaped to keep the wrist neutral, with adjustable DPI, silent click switches, and rechargeable USB-C battery."),
    ("prod-elec-portable-ssd-2tb",              "Rugged 2 terabyte portable USB-C solid state drive with shock-resistant aluminum housing, up to 1050 MB/s read speeds, and hardware AES encryption."),
    ("prod-elec-wireless-router-mesh",          "Tri-band Wi-Fi 6E mesh router system that blankets large homes in fast wireless internet, with parental controls and seamless roaming across nodes."),
    ("prod-elec-webcam-1080p",                  "Plug-and-play 1080p HD streaming webcam with stereo microphones, autofocus, and a privacy shutter for video conferencing and online meetings."),
    ("prod-elec-smart-doorbell-camera",         "Battery-powered smart video doorbell camera with motion detection, two-way talk, and crystal clear night vision streamed to your smartphone."),
    ("prod-elec-bluetooth-speaker-waterproof",  "Waterproof portable Bluetooth party speaker with 360-degree sound, deep bass radiator, and 24-hour playtime for pool, beach, and backyard gatherings."),
    ("prod-elec-vr-headset",                    "Standalone virtual reality headset with high-resolution per-eye displays, inside-out tracking, and a library of immersive VR games and fitness apps."),
    ("prod-elec-dash-cam-4k",                   "4K front and 1080p rear dash camera with GPS logging, parking surveillance mode, and wide dynamic range sensor for clear day and night recording."),
    ("prod-elec-graphics-tablet",               "Pen display graphics tablet with 8192 levels of pressure sensitivity, programmable shortcut keys, and color-accurate screen for digital illustrators and photo editors."),
    ("prod-elec-usb-c-hub",                     "11-in-1 USB-C docking station hub with dual HDMI 4K output, gigabit Ethernet, SD card reader, and 100W passthrough charging for laptops."),
    ("prod-elec-air-fryer-toaster-oven",        "Smart air fryer toaster oven combo with 12 cooking modes, convection technology, and a touchscreen display that crisps food using little to no oil."),
]
kitchen_extras = [
    ("prod-kitchen-blender-vitamix",            "High-performance 1200 watt countertop blender with stainless steel laser-cut blades, variable speed control, and tritan jug for smoothies, soups, and nut butters."),
    ("prod-kitchen-cast-iron-skillet",          "Pre-seasoned 12 inch cast iron skillet with helper handle and pour spouts; oven, stovetop, grill, and campfire safe for decades of cooking."),
    ("prod-kitchen-chef-knife",                 "8 inch high-carbon stainless steel chef knife with full tang triple-riveted ergonomic handle, hand sharpened to a razor edge for everyday slicing and dicing."),
    ("prod-kitchen-instant-pot",                "9-in-1 multi-cooker pressure cooker with slow cook, rice, yogurt, saute, and sterilize programs, suitable for one-pot family dinners in under an hour."),
    ("prod-kitchen-stand-mixer",                "Tilt-head 5 quart stand mixer with planetary mixing action and 10 attachments for kneading sourdough, whipping meringue, and grinding meat in a single appliance."),
    ("prod-kitchen-dutch-oven",                 "Enameled cast iron 7 quart Dutch oven with self-basting lid, perfect for braising, baking artisan bread, and simmering hearty stews on stove or oven."),
    ("prod-kitchen-rice-cooker-japanese",       "Japanese induction rice cooker with fuzzy logic, multiple grain settings, and a non-stick spherical inner pot that delivers fluffy white, brown, or sushi rice."),
    ("prod-kitchen-coffee-grinder-burr",        "Conical burr coffee grinder with 40 grind settings, suitable for pour-over, French press, drip, and even espresso brewing with consistent particle size."),
    ("prod-kitchen-french-press",               "8 cup double-walled stainless steel French press coffee maker with reusable mesh filter and heat-retaining vacuum insulation."),
    ("prod-kitchen-electric-kettle-temp",       "Variable temperature gooseneck electric kettle with one-degree precision settings, ideal for tea brewing and pour-over coffee enthusiasts."),
    ("prod-kitchen-cutting-board-walnut",       "End-grain walnut wood cutting board with juice groove and built-in handles, gentle on knife edges and gorgeous as a charcuterie serving board."),
    ("prod-kitchen-sous-vide-circulator",       "Immersion sous vide precision circulator with smartphone app control, achieving restaurant-quality results for steak, eggs, and vegetables at home."),
    ("prod-kitchen-glass-storage-set",          "20-piece airtight borosilicate glass meal-prep food storage container set with leak-proof bamboo lids, freezer, microwave, and dishwasher safe."),
    ("prod-kitchen-pizza-stone-cordierite",     "15 inch round cordierite pizza baking stone for ovens and grills, delivering crispy artisan crusts and even browning on homemade pies."),
]
fitness_extras = [
    ("prod-fit-yoga-mat-eco",                   "Extra thick 6mm eco-friendly TPE yoga mat with non-slip dual-textured surface and carrying strap, ideal for vinyasa flow, Pilates, and home workouts."),
    ("prod-fit-adjustable-dumbbells",           "Pair of adjustable dumbbells with rapid dial system, replacing 15 weights from 5 to 52 pounds for compact strength training at home."),
    ("prod-fit-jump-rope-speed",                "Speed jump rope with ball-bearing handles, adjustable steel cable, and free workout app, perfect for high-intensity cardio and double-unders."),
    ("prod-fit-resistance-bands-set",           "11-piece resistance band set with stackable tubes up to 150 pounds, door anchor, handles, and ankle straps for total-body resistance training."),
    ("prod-fit-foam-roller-vibrating",          "Vibrating foam roller with five intensity settings for deep tissue myofascial release, post-workout muscle recovery, and mobility work."),
    ("prod-fit-treadmill-folding",              "Folding electric treadmill with cushioned running deck, 12 incline levels, and built-in heart rate sensors for cardio sessions in small apartments."),
    ("prod-fit-rowing-machine-magnetic",        "Magnetic resistance rowing machine with 16 levels, performance monitor, and aluminum rail for smooth full-body indoor rowing workouts."),
    ("prod-fit-cycling-shoes-spd",              "Indoor cycling shoes with stiff carbon-reinforced sole, breathable mesh, and dual SPD cleat compatibility for spin classes and road bikes."),
    ("prod-fit-trail-running-shoes",            "Aggressive lugged trail running shoes with rock plate and waterproof membrane, designed for muddy ultra-marathon courses and technical mountain terrain."),
    ("prod-fit-walking-shoes-comfort",          "Cushioned all-day walking shoes with memory foam insole, arch support, and slip-resistant outsole for nurses, teachers, and city walkers."),
    ("prod-fit-tennis-racket",                  "Graphite composite tennis racket with 100 square inch head, balanced 300 gram frame, and shock-absorbing grip for intermediate club players."),
    ("prod-fit-bike-helmet-aero",               "Aerodynamic road cycling helmet with MIPS rotational impact protection, magnetic visor, and adjustable retention dial for triathletes and roadies."),
    ("prod-fit-water-bottle-insulated",         "24 oz double-walled vacuum-insulated stainless steel water bottle that keeps drinks cold for 24 hours and hot for 12, with leak-proof sport lid."),
    ("prod-fit-smartwatch-running",             "GPS multisport smartwatch with VO2 max estimation, training load tracking, optical heart rate, and breadcrumb navigation for runners and triathletes."),
]
books_extras = [
    ("prod-books-mystery-detective",            "Gripping noir detective mystery novel set in 1940s San Francisco, where a hard-boiled private investigator unravels a deadly conspiracy through fog-soaked streets."),
    ("prod-books-romance-historical",           "Sweeping historical romance novel set in Regency-era England, following a clever spinster and a brooding duke as their forbidden love defies society."),
    ("prod-books-scifi-space-opera",            "Hugo Award-nominated science fiction space opera featuring rebel starships, alien diplomacy, and quantum teleportation across a galactic empire on the brink of civil war."),
    ("prod-books-biography-einstein",           "Critically acclaimed biography of Albert Einstein chronicling his Bohemian youth, miracle year, and lifelong quest for a unified field theory of physics."),
    ("prod-books-cookbook-italian",             "Cookbook celebrating regional Italian home cooking with 200 recipes for fresh pasta, ragu, focaccia, tiramisu, and seasonal vegetable contorni."),
    ("prod-books-self-help-habits",             "Bestselling self-help guide to building tiny daily habits that compound into life-changing results, packed with neuroscience-backed strategies and templates."),
    ("prod-books-childrens-picturebook",        "Heartwarming children's picture book about a curious little fox who learns to share, beautifully illustrated in watercolor for ages three to six."),
    ("prod-books-history-rome",                 "Sweeping narrative history of the rise and fall of the Roman Empire, from the Republic through Constantine, drawing on the latest archaeological evidence."),
    ("prod-books-thriller-spy",                 "Pulse-pounding espionage thriller in which a disavowed CIA operative races across European capitals to stop a rogue intelligence cell from triggering nuclear war."),
    ("prod-books-philosophy-stoicism",          "Accessible modern introduction to Stoic philosophy, applying lessons from Marcus Aurelius and Epictetus to anxiety, productivity, and meaningful living."),
    ("prod-books-poetry-contemporary",          "Award-winning contemporary poetry collection exploring grief, immigration, and rediscovered joy through lyrical free verse and prose poems."),
    ("prod-books-business-startup",             "Practical handbook for founding a tech startup, covering customer discovery, product-market fit, fundraising, and scaling teams from zero to one hundred employees."),
    ("prod-books-graphic-novel-superhero",      "Beautifully illustrated graphic novel reinventing a classic superhero mythos with morally complex villains, sweeping splash pages, and meta-textual storytelling."),
    ("prod-books-young-adult-dystopia",         "Young adult dystopian trilogy opener about a teenage rebel surviving in a flooded post-climate-collapse mega-city ruled by an authoritarian corporate council."),
]

for pid, desc in electronics_extras:
    products.append({"id": pid, "category": "electronics", "description": desc})
for pid, desc in kitchen_extras:
    products.append({"id": pid, "category": "kitchen", "description": desc})
for pid, desc in fitness_extras:
    products.append({"id": pid, "category": "fitness", "description": desc})
for pid, desc in books_extras:
    products.append({"id": pid, "category": "books", "description": desc})

assert len(products) == 60, f"Expected 60 products, got {len(products)}"

os.makedirs("/home/user/myproject", exist_ok=True)
with open("/home/user/myproject/products.json", "w") as f:
    json.dump(products, f, indent=2)
print(f"Seeded {len(products)} products to /home/user/myproject/products.json")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="Voyage AI voyage-3 embeddings + LanceDB semantic product-search task environment."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4,
#                    pandas==2.2.3, voyageai>=0.3,<0.5
#   - /home/user/myproject/products.json: 60-row deterministic English product catalogue
#     with 4 anchor ids (prod-electronics-headphones-anc, prod-kitchen-espresso-machine,
#     prod-fitness-carbon-marathon-shoes, prod-books-dragon-wizard-fantasy)
