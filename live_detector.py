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

    import collections
    from collections import Counter

    # Session Stats
    stats = {
        'total': 0,
        'recyclables': 0,
        'compostables': 0,
    }
    
    # Prediction history queue for temporal smoothing (prevents flickers)
    predictions_queue = collections.deque(maxlen=10)
    last_stable_class = None
    
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

        # Update prediction smoothing queue
        if res is not None:
            # If the model has reasonable confidence, add it. Otherwise, add None (empty zone)
            if res['confidence'] > 0.55:
                predictions_queue.append(res['class'])
            else:
                predictions_queue.append(None)
        else:
            predictions_queue.append(None)

        # Get majority class in prediction window
        smoothed_class = None
        if len(predictions_queue) > 0:
            counter = Counter(predictions_queue)
            most_common = counter.most_common(1)
            # Only count as a valid class if it's the majority and not None
            if most_common and most_common[0][0] is not None:
                # We require at least 4 out of 10 frames to agree on a class to stabilize
                if most_common[0][1] >= 4:
                    smoothed_class = most_common[0][0]

        # Update stats when a new stable detection is confirmed
        if smoothed_class is not None:
            if smoothed_class != last_stable_class:
                meta = WASTE_METADATA.get(smoothed_class, WASTE_METADATA["miscellaneous trash"])
                stats['total'] += 1
                if meta['recyclable']:
                    stats['recyclables'] += 1
                if smoothed_class in ["food organics", "vegetation"]:
                    stats['compostables'] += 1
                
                last_stable_class = smoothed_class
        else:
            # Reset stable tracker once the zone is empty (at least 6 frames are None)
            if predictions_queue.count(None) >= 6:
                last_stable_class = None

        # Try to get window size dynamically
        try:
            rect = cv2.getWindowImageRect("EcoSort AI - Waste Classifier Dashboard")
            if rect is not None and rect[2] > 300 and rect[3] > 300:
                win_w, win_h = rect[2], rect[3]
            else:
                win_w, win_h = WIN_W, WIN_H
        except Exception:
            win_w, win_h = WIN_W, WIN_H

        # --- BUILD FUTURISTIC CANVAS ---
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        canvas[:, :] = BG_COLOR
        
        # 1. Header Bar
        cv2.rectangle(canvas, (0, 0), (win_w, 80), (16, 10, 8), -1)
        cv2.line(canvas, (0, 80), (win_w, 80), BORDER_COLOR, 1)
        cv2.putText(canvas,
                    "ECOSORT AI - WASTE CLASSIFICATION CONSOLE",
                    ((win_w - 680) // 2, 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.85,
                    HUD_GOLD,
                    2,
                    cv2.LINE_AA)
        
        # Right Sidebar Panel dimensions
        panel_w = 480
        panel_x1 = win_w - panel_w - 40
        panel_x2 = win_w - 40
        panel_y1 = 110
        panel_y2 = win_h - 50

        # Draw Right Panel Background
        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), PANEL_COLOR, -1)
        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), BORDER_COLOR, 1)

        # Available space for camera layout
        avail_w = panel_x1 - 80
        avail_h = win_h - 160
        
        # Aspect-ratio preserving scale
        scale = min(avail_w / CAM_W, avail_h / CAM_H)
        cam_draw_w = int(CAM_W * scale)
        cam_draw_h = int(CAM_H * scale)
        
        # Center camera frame inside left available space
        cam_draw_x = 40 + (avail_w - cam_draw_w) // 2
        cam_draw_y = 110 + (avail_h - cam_draw_h) // 2

        # 2. Draw Camera feed into canvas
        resized_frame = cv2.resize(frame, (cam_draw_w, cam_draw_h))
        canvas[cam_draw_y:cam_draw_y + cam_draw_h, cam_draw_x:cam_draw_x + cam_draw_w] = resized_frame
        
        # Neon Border around camera viewport
        cv2.rectangle(canvas, (cam_draw_x - 2, cam_draw_y - 2), (cam_draw_x + cam_draw_w + 2, cam_draw_y + cam_draw_h + 2), HUD_GOLD, 2)
        cv2.putText(canvas, "LIVE SCANNER FEED", (cam_draw_x, cam_draw_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_GOLD, 1, cv2.LINE_AA)

        # 3. Draw Scanning Zone Box inside camera viewport (Absolute coordinates on canvas)
        abs_x1 = cam_draw_x + int(ZONE_X1 * scale)
        abs_y1 = cam_draw_y + int(ZONE_Y1 * scale)
        abs_x2 = cam_draw_x + int(ZONE_X2 * scale)
        abs_y2 = cam_draw_y + int(ZONE_Y2 * scale)
        
        # Color pulsing vignette/bracket depending on if target detected
        bracket_color = (0, 255, 0)  # Default green
        if smoothed_class is not None:
            meta = WASTE_METADATA.get(smoothed_class, WASTE_METADATA["miscellaneous trash"])
            bracket_color = meta["color"]
            
        # Draw nice corner brackets for the scanning zone
        length = int(25 * scale)
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
                    0.45 * min(scale, 1.5),
                    bracket_color,
                    1,
                    cv2.LINE_AA)

        # Interactive guidance hint text under Scanning Zone
        if smoothed_class is not None:
            hint_text = "SCAN SUCCESSFUL! ROUTE TO BIN."
            hint_color = (0, 255, 0) # Bright green
        elif res is not None and 0.20 < res['confidence'] <= 0.55:
            hint_text = "HINT: HOLD ITEM CLOSER & STEADY"
            hint_color = (0, 165, 255) # Amber orange
        else:
            hint_text = "PLACE ITEM STABLE INSIDE SCAN BOX"
            hint_color = (255, 230, 0) # Neon blue/cyan

        cv2.putText(canvas,
                    hint_text,
                    (abs_x1 - 15, abs_y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * min(scale, 1.5),
                    hint_color,
                    1,
                    cv2.LINE_AA)

        # 4. Draw Right Panel Console
        # Set default values if no prediction yet
        display_title = "AWAITING MATERIAL"
        display_bin = "ROUTE TO: --"
        display_instructions = ["Please place a waste item", "inside the detection zone box", "on the left to classify."]
        display_impact = "Sorting waste properly diverts recyclable materials from reaching landfills."
        display_color = BORDER_COLOR
        conf_percentage = 0.0
        
        if smoothed_class is not None:
            meta = WASTE_METADATA.get(smoothed_class, WASTE_METADATA["miscellaneous trash"])
            display_title = meta["title"].upper()
            display_bin = f"ROUTE TO: {meta['bin']}"
            display_instructions = meta["instructions"]
            display_impact = meta["impact"]
            display_color = meta["color"]
            if res and res['class'] == smoothed_class:
                conf_percentage = res['confidence'] * 100.0
            else:
                conf_percentage = 85.0 # default stabilized display confidence

        # Classification Header Box
        cv2.rectangle(canvas, (panel_x1 + 20, panel_y1 + 20), (panel_x2 - 20, panel_y1 + 70), display_color, -1)
        cv2.putText(canvas,
                    display_title,
                    (panel_x1 + 40, panel_y1 + 53),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    TEXT_WHITE,
                    2,
                    cv2.LINE_AA)

        # Confidence Bar Indicator
        cv2.putText(canvas,
                    f"CONFIDENCE: {conf_percentage:.1f}%" + (" (Sandbox)" if res and res.get('is_mock') else ""),
                    (panel_x1 + 20, panel_y1 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    TEXT_WHITE,
                    1,
                    cv2.LINE_AA)
        
        # Outer bar
        cv2.rectangle(canvas, (panel_x1 + 20, panel_y1 + 112), (panel_x2 - 20, panel_y1 + 127), (70, 70, 70), -1)
        # Inner bar
        bar_fill = int((conf_percentage / 100.0) * (panel_w - 40))
        if bar_fill > 0:
            cv2.rectangle(canvas, (panel_x1 + 20, panel_y1 + 112), (panel_x1 + 20 + bar_fill, panel_y1 + 127), display_color, -1)

        # Routing Destination
        cv2.rectangle(canvas, (panel_x1 + 20, panel_y1 + 152), (panel_x2 - 20, panel_y1 + 192), display_color, 2)
        cv2.putText(canvas,
                    display_bin,
                    (panel_x1 + 35, panel_y1 + 177),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    display_color,
                    1,
                    cv2.LINE_AA)

        # Sorting Guidelines Section
        y_section = panel_y1 + 230
        cv2.putText(canvas, "SORTING GUIDELINES", (panel_x1 + 20, y_section), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_GOLD, 1, cv2.LINE_AA)
        cv2.line(canvas, (panel_x1 + 20, y_section + 5), (panel_x2 - 20, y_section + 5), BORDER_COLOR, 1)
        
        # Bullet list coordinates
        bullet_list = [f"- {inst}" for inst in display_instructions]
        draw_multiline_text(canvas, bullet_list, (panel_x1 + 20, y_section + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, 25)

        # Telemetry Stats Section
        y_telemetry = panel_y1 + 355
        cv2.putText(canvas, "SESSION TELEMETRY", (panel_x1 + 20, y_telemetry), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_GOLD, 1, cv2.LINE_AA)
        cv2.line(canvas, (panel_x1 + 20, y_telemetry + 5), (panel_x2 - 20, y_telemetry + 5), BORDER_COLOR, 1)

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
            cv2.putText(canvas, f"{name}:", (panel_x1 + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (panel_x2 - 100, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, HUD_GOLD, 1, cv2.LINE_AA)

        # Eco Impact Box (translucent background box)
        y_impact = panel_y2 - 115
        cv2.rectangle(canvas, (panel_x1 + 20, y_impact), (panel_x2 - 20, panel_y2 - 20), (20, 45, 25) if smoothed_class else (16, 20, 28), -1)
        cv2.rectangle(canvas, (panel_x1 + 20, y_impact), (panel_x2 - 20, panel_y2 - 20), (129, 185, 16) if smoothed_class else BORDER_COLOR, 1)
        
        cv2.putText(canvas, "ECO-IMPACT FACT:", (panel_x1 + 35, y_impact + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, (129, 185, 16) if smoothed_class else HUD_GOLD, 1, cv2.LINE_AA)
        
        # Wrap the impact fact text to fit
        wrapped_lines = wrap_text(display_impact, max_chars=48)
        draw_multiline_text(canvas, wrapped_lines, (panel_x1 + 35, y_impact + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, 18)

        # Status Footer Bar
        cv2.rectangle(canvas, (0, win_h - 30), (win_w, win_h), (10, 8, 6), -1)
        cv2.putText(canvas,
                    f"Model: {status_msg}  |  Controls: 'r' to Reset stats, 'q' to Quit",
                    (20, win_h - 10),
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
            last_stable_class = None
            predictions_queue.clear()
            print("Session statistics reset!")

    # Cleanup
    running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully.")

if __name__ == "__main__":
    main()
