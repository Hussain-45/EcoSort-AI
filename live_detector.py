import os
import time
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Constants and Settings
MODEL_PATH = "best_waste_model.pth"
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

# UI Dimensions
WIN_W = 1280
WIN_H = 720
CAM_W = 640
CAM_H = 480

# Camera placement relative offset on canvas
CAM_X = 40
CAM_Y = 140

# Central Scanning Zone size and coordinates (relative to camera frame)
ZONE_SIZE = 300
ZONE_X1 = (CAM_W - ZONE_SIZE) // 2
ZONE_Y1 = (CAM_H - ZONE_SIZE) // 2
ZONE_X2 = ZONE_X1 + ZONE_SIZE
ZONE_Y2 = ZONE_Y1 + ZONE_SIZE

# Metadata for waste categories (matching Streamlit, containing BGR colors)
WASTE_METADATA = {
    "cardboard": {
        "title": "Cardboard",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "color": (246, 130, 59),  # BGR representation of #3b82f6 (approx)
        "instructions": [
            "Flatten boxes completely.",
            "Remove packing tape/straps.",
            "Throw greasy boxes in compost."
        ],
        "impact": "Recycling 1 ton of cardboard saves 46 gallons of oil."
    },
    "food organics": {
        "title": "Food Organics",
        "recyclable": True,
        "bin": "Compost Bin (Green)",
        "color": (129, 185, 16),  # BGR representation of #10b981 (approx)
        "instructions": [
            "Scraps, leftovers, and tea bags.",
            "Remove plastic stickers/wraps.",
            "Towels soiled with food are OK."
        ],
        "impact": "Composting redirects organics to create nutrient-rich soil."
    },
    "glass": {
        "title": "Glass Bottles & Jars",
        "recyclable": True,
        "bin": "Glass / Recycle Bin (Blue)",
        "color": (8, 179, 234),  # BGR representation of #eab308 (approx)
        "instructions": [
            "Empty and rinse all jars.",
            "Remove metal caps and rings.",
            "Do not mix drinking glasses/mirrors."
        ],
        "impact": "Glass is 100% recyclable and can be recycled endlessly."
    },
    "metal": {
        "title": "Metal & Cans",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "color": (246, 130, 59),  # BGR representation of #3b82f6
        "instructions": [
            "Rinse soda and soup cans.",
            "Crumple clean foil into a ball.",
            "No need to remove paper labels."
        ],
        "impact": "Recycled aluminum uses 95% less energy than raw bauxite."
    },
    "miscellaneous trash": {
        "title": "General Trash",
        "recyclable": False,
        "bin": "Landfill Bin (Black/Gray)",
        "color": (128, 128, 128),  # Gray
        "instructions": [
            "Wipes, Styrofoam, and chip bags.",
            "Ensure items are dry.",
            "Minimizing this bin is our goal."
        ],
        "impact": "Landfill waste stays buried for centuries. Reduce it!"
    },
    "paper": {
        "title": "Mixed Paper",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "color": (246, 130, 59),  # BGR representation of #3b82f6
        "instructions": [
            "Notebooks, flyers, and bags.",
            "Wet paper cannot be recycled.",
            "Shredded paper must be bagged."
        ],
        "impact": "Recycling one ton of paper saves 17 trees."
    },
    "plastic": {
        "title": "Recyclable Plastics",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "color": (246, 130, 59),  # BGR representation of #3b82f6
        "instructions": [
            "Rinse plastic bottles & tubs.",
            "Crush bottles to save space.",
            "Grocery bags go to store drop-off."
        ],
        "impact": "Plastic takes 500 years to decompose. Recycle it!"
    },
    "textile trash": {
        "title": "Textile Trash / Fabric",
        "recyclable": False,
        "bin": "Fabric Recycle / Donation",
        "color": (246, 92, 139),  # BGR representation of #8b5cf6 (approx)
        "instructions": [
            "Donate wearable clothing.",
            "Cut unwearable rags for cleanup.",
            "Fabric recycling bins take rest."
        ],
        "impact": "85% of textiles end up in landfills. Repurpose fabrics."
    },
    "vegetation": {
        "title": "Yard Waste",
        "recyclable": True,
        "bin": "Yard Waste Bin (Green)",
        "color": (129, 185, 16),  # BGR representation of #10b981
        "instructions": [
            "Leaves, twigs, grass clippings.",
            "No soil, rocks, or treated wood.",
            "Place loose or in paper bags."
        ],
        "impact": "Yard waste is composted for agricultural mulch."
    }
}

# Image normalization transforms (must match train.py)
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Shared Thread State
shared_frame_crop = None
shared_results = None
shared_frame_flag = False
thread_lock = threading.Lock()
running = True

# 3. Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing EcoSort Model on {device}...")

model = None
status_msg = ""
if os.path.exists(MODEL_PATH):
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        status_msg = f"Trained ResNet18 Weights ({str(device).upper()})"
        print("Model loaded successfully!")
    except Exception as e:
        status_msg = f"Failed to load weights: {e}"
        print(f"Error loading weights: {e}")
else:
    status_msg = "No trained weights found (Running Sandbox Mode)"
    print("WARNING: best_waste_model.pth not found! Running in sandbox mode.")

# 4. Background Prediction Thread Worker
def inference_worker():
    global shared_results, shared_frame_flag
    print("Background inference worker thread started.")
    
    while running:
        crop_to_process = None
        
        # Check if new frame crop is available
        with thread_lock:
            if shared_frame_flag and shared_frame_crop is not None:
                crop_to_process = shared_frame_crop.copy()
                shared_frame_flag = False
                
        if crop_to_process is not None:
            if model is not None:
                try:
                    # OpenCV BGR -> PyTorch RGB format conversion
                    rgb_crop = cv2.cvtColor(crop_to_process, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_crop)
                    tensor = transform_pipeline(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        confidence, pred_idx = torch.max(probs, dim=0)
                        
                        pred_class = CLASS_NAMES[pred_idx.item()].lower()
                        conf_val = confidence.item()
                        
                        with thread_lock:
                            shared_results = {
                                'class': pred_class,
                                'confidence': conf_val,
                                'is_mock': False,
                                'timestamp': time.time()
                            }
                except Exception as e:
                    print(f"Worker Inference error: {e}")
            else:
                # Sandbox Mock mode
                time.sleep(0.15)  # Simulate CPU latency
                import random
                mock_class = random.choice(list(WASTE_METADATA.keys()))
                with thread_lock:
                    shared_results = {
                        'class': mock_class,
                        'confidence': 0.89,
                        'is_mock': True,
                        'timestamp': time.time()
                    }
        else:
            # Idle sleep to prevent high CPU utilization
            time.sleep(0.01)

# Start prediction thread
worker_thread = threading.Thread(target=inference_worker, daemon=True)
worker_thread.start()

# 5. UI Helper Functions
def draw_multiline_text(img, text_list, start_pt, font, scale, color, thickness, line_spacing=25):
    x, y = start_pt
    for i, line in enumerate(text_list):
        cv2.putText(img, line, (x, y + i * line_spacing), font, scale, color, thickness, cv2.LINE_AA)

def wrap_text(text, max_chars=36):
    words = text.split(' ')
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(' '.join(current_line))
    return lines

# 6. Main OpenCV Loop
def main():
    global shared_frame_crop, shared_frame_flag, running
    
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera device.")
        return

    # Window settings
    cv2.namedWindow("EcoSort AI - Waste Classifier Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EcoSort AI - Waste Classifier Dashboard", WIN_W, WIN_H)

    # Session Stats
    stats = {
        'total': 0,
        'recyclables': 0,
        'compostables': 0,
    }
    
    last_processed_class = None
    last_processed_time = 0
    
    # UI Design Palette (BGR)
    BG_COLOR = (25, 15, 11)        # Deep slate/navy
    PANEL_COLOR = (59, 41, 30)     # Muted slate panel
    BORDER_COLOR = (80, 80, 80)    # Gray borders
    HUD_GOLD = (0, 230, 255)       # Cyberpunk Gold
    TEXT_WHITE = (240, 240, 240)
    TEXT_MUTED = (180, 180, 180)
    
    print("\n" + "="*50)
    print(" EcoSort AI Webcam Interface is Ready!")
    print("  - Place waste item inside the green box to scan.")
    print("  - Controls:")
    print("    'r' : Reset session statistics")
    print("    'q' / ESC : Quit the application")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        # Mirror frame for natural webcam look
        frame = cv2.flip(frame, 1)
        
        # Resize frame to standard 640x480 for consistent layout placement
        frame = cv2.resize(frame, (CAM_W, CAM_H))
        
        # Get crop region
        crop = frame[ZONE_Y1:ZONE_Y2, ZONE_X1:ZONE_X2]
        
        # Push crop to background prediction thread
        with thread_lock:
            shared_frame_crop = crop.copy()
            shared_frame_flag = True
            
        # Retrieve latest prediction results
        res = None
        with thread_lock:
            if shared_results is not None:
                res = shared_results.copy()

        # Update stats when a new stable detection is confirmed
        if res is not None and res['confidence'] > 0.65:
            detected_class = res['class']
            current_time = time.time()
            
            # Count only if it is a new class or some time has passed to prevent double counting
            if detected_class != last_processed_class or (current_time - last_processed_time) > 4.0:
                meta = WASTE_METADATA.get(detected_class, WASTE_METADATA["miscellaneous trash"])
                stats['total'] += 1
                if meta['recyclable']:
                    stats['recyclables'] += 1
                if detected_class in ["food organics", "vegetation"]:
                    stats['compostables'] += 1
                
                last_processed_class = detected_class
                last_processed_time = current_time

        # --- BUILD FUTURISTIC CANVAS ---
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:, :] = BG_COLOR
        
        # 1. Header Bar
        cv2.rectangle(canvas, (0, 0), (WIN_W, 80), (16, 10, 8), -1)
        cv2.line(canvas, (0, 80), (WIN_W, 80), BORDER_COLOR, 1)
        cv2.putText(canvas,
                    "ECOSORT AI - WASTE CLASSIFICATION CONSOLE",
                    (WIN_W // 2 - 340, 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.85,
                    HUD_GOLD,
                    2,
                    cv2.LINE_AA)
        
        # 2. Draw Camera feed into canvas
        canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = frame
        
        # Neon Border around camera viewport
        cv2.rectangle(canvas, (CAM_X - 2, CAM_Y - 2), (CAM_X + CAM_W + 2, CAM_Y + CAM_H + 2), HUD_GOLD, 2)
        cv2.putText(canvas, "LIVE SCANNER FEED", (CAM_X, CAM_Y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_GOLD, 1, cv2.LINE_AA)

        # 3. Draw Scanning Zone Box inside camera viewport (Absolute coordinates on canvas)
        abs_x1 = CAM_X + ZONE_X1
        abs_y1 = CAM_Y + ZONE_Y1
        abs_x2 = CAM_X + ZONE_X2
        abs_y2 = CAM_Y + ZONE_Y2
        
        # Color pulsing vignette/bracket depending on if target detected
        bracket_color = (0, 255, 0)  # Default green
        if res is not None and res['confidence'] > 0.70:
            meta = WASTE_METADATA.get(res['class'], WASTE_METADATA["miscellaneous trash"])
            bracket_color = meta["color"]
            
        # Draw nice corner brackets for the scanning zone
        length = 25
        # Top-Left
        cv2.line(canvas, (abs_x1, abs_y1), (abs_x1 + length, abs_y1), bracket_color, 3, cv2.LINE_AA)
        cv2.line(canvas, (abs_x1, abs_y1), (abs_x1, abs_y1 + length), bracket_color, 3, cv2.LINE_AA)
        # Top-Right
        cv2.line(canvas, (abs_x2, abs_y1), (abs_x2 - length, abs_y1), bracket_color, 3, cv2.LINE_AA)
        cv2.line(canvas, (abs_x2, abs_y1), (abs_x2, abs_y1 + length), bracket_color, 3, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(canvas, (abs_x1, abs_y2), (abs_x1 + length, abs_y2), bracket_color, 3, cv2.LINE_AA)
        cv2.line(canvas, (abs_x1, abs_y2), (abs_x1, abs_y2 - length), bracket_color, 3, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(canvas, (abs_x2, abs_y2), (abs_x2 - length, abs_y2), bracket_color, 3, cv2.LINE_AA)
        cv2.line(canvas, (abs_x2, abs_y2), (abs_x2, abs_y2 - length), bracket_color, 3, cv2.LINE_AA)
        
        # Text above scanning zone
        cv2.putText(canvas,
                    "WASTE DETECTION ZONE",
                    (abs_x1, abs_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    bracket_color,
                    1,
                    cv2.LINE_AA)

        # 4. Draw Right Panel Console
        cv2.rectangle(canvas, (720, 110), (1240, 680), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (720, 110), (1240, 680), BORDER_COLOR, 1)

        # Set default values if no prediction yet
        display_title = "AWAITING MATERIAL"
        display_bin = "ROUTE TO: --"
        display_instructions = ["Please place a waste item", "inside the detection zone box", "on the left to classify."]
        display_impact = "Sorting waste properly diverts recyclable materials from reaching landfills."
        display_color = BORDER_COLOR
        conf_percentage = 0.0
        
        if res is not None:
            meta = WASTE_METADATA.get(res['class'], WASTE_METADATA["miscellaneous trash"])
            display_title = meta["title"].upper()
            display_bin = f"ROUTE TO: {meta['bin']}"
            display_instructions = meta["instructions"]
            display_impact = meta["impact"]
            display_color = meta["color"]
            conf_percentage = res['confidence'] * 100.0

        # Classification Header Box
        cv2.rectangle(canvas, (740, 130), (1220, 180), display_color, -1)
        cv2.putText(canvas,
                    display_title,
                    (760, 163),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    TEXT_WHITE,
                    2,
                    cv2.LINE_AA)

        # Confidence Bar Indicator
        cv2.putText(canvas,
                    f"CONFIDENCE: {conf_percentage:.1f}%" + (" (Sandbox)" if res and res.get('is_mock') else ""),
                    (740, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    TEXT_WHITE,
                    1,
                    cv2.LINE_AA)
        
        # Outer bar
        cv2.rectangle(canvas, (740, 222), (1220, 237), (70, 70, 70), -1)
        # Inner bar
        bar_fill = int((conf_percentage / 100.0) * 480)
        if bar_fill > 0:
            cv2.rectangle(canvas, (740, 222), (740 + bar_fill, 237), display_color, -1)

        # Routing Destination
        cv2.rectangle(canvas, (740, 262), (1220, 302), display_color, 2)
        cv2.putText(canvas,
                    display_bin,
                    (755, 287),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    display_color,
                    1,
                    cv2.LINE_AA)

        # Sorting Guidelines Section
        y_section = 340
        cv2.putText(canvas, "SORTING GUIDELINES", (740, y_section), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_GOLD, 1, cv2.LINE_AA)
        cv2.line(canvas, (740, y_section + 5), (1220, y_section + 5), BORDER_COLOR, 1)
        
        # Bullet list coordinates
        bullet_list = [f"- {inst}" for inst in display_instructions]
        draw_multiline_text(canvas, bullet_list, (740, y_section + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, 25)

        # Telemetry Stats Section
        y_telemetry = 455
        cv2.putText(canvas, "SESSION TELEMETRY", (740, y_telemetry), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_GOLD, 1, cv2.LINE_AA)
        cv2.line(canvas, (740, y_telemetry + 5), (1220, y_telemetry + 5), BORDER_COLOR, 1)

        # Calculations
        div_rate = 0
        if stats['total'] > 0:
            div_rate = int(((stats['recyclables'] + stats['compostables']) / stats['total']) * 100)

        # Draw metrics
        metrics = [
            ("Total Items Scanned", f"{stats['total']}"),
            ("Recyclable Items", f"{stats['recyclables']}"),
            ("Organics Compostable", f"{stats['compostables']}"),
            ("Landfill Diversion Rate", f"{div_rate}%")
        ]

        for i, (name, val) in enumerate(metrics):
            y_pos = y_telemetry + 25 + i * 20
            cv2.putText(canvas, f"{name}:", (740, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (1080, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, HUD_GOLD, 1, cv2.LINE_AA)

        # Eco Impact Box (translucent background box)
        y_impact = 565
        cv2.rectangle(canvas, (740, y_impact), (1220, y_impact + 95), (20, 45, 25) if res else (16, 20, 28), -1)
        cv2.rectangle(canvas, (740, y_impact), (1220, y_impact + 95), (129, 185, 16) if res else BORDER_COLOR, 1)
        
        cv2.putText(canvas, "ECO-IMPACT FACT:", (755, y_impact + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, (129, 185, 16) if res else HUD_GOLD, 1, cv2.LINE_AA)
        
        # Wrap the impact fact text to fit
        wrapped_lines = wrap_text(display_impact, max_chars=48)
        draw_multiline_text(canvas, wrapped_lines, (755, y_impact + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, 18)

        # Status Footer Bar
        cv2.rectangle(canvas, (0, WIN_H - 30), (WIN_W, WIN_H), (10, 8, 6), -1)
        cv2.putText(canvas,
                    f"Model: {status_msg}  |  Controls: 'r' to Reset stats, 'q' to Quit",
                    (20, WIN_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    TEXT_MUTED,
                    1,
                    cv2.LINE_AA)

        # Render complete canvas to dashboard window
        cv2.imshow("EcoSort AI - Waste Classifier Dashboard", canvas)
        
        # Keyboard polling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC is 27
            break
        elif key == ord('r'):
            stats['total'] = 0
            stats['recyclables'] = 0
            stats['compostables'] = 0
            last_processed_class = None
            print("Session statistics reset!")

    # Cleanup
    running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully.")

if __name__ == "__main__":
    main()
