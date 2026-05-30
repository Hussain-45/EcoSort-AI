# EcoSort AI - Real-World Waste Classification Console

EcoSort AI is a real-time, desktop-based waste classification system designed for industrial and household sorting. It leverages deep learning (PyTorch ResNet18) and computer vision (OpenCV) to identify waste items, recommend recycling or compost routes, and display detailed sorting guidelines on a futuristic, high-performance dashboard.

---

## 🔍 The Problem & The Solution

### 1. The Dataset Problem (Clean vs. Real Trash)
* **The Issue:** Most garbage detection models are trained on clean, pristine studio images of plastic bottles, tin cans, or boxes against clean white backgrounds. When deployed in the real world (or on sorting conveyor belts), these models fail completely because actual trash is crumpled, squashed, dirty, torn, or partially degraded.
* **The Solution:** EcoSort AI is fine-tuned on the **RealWaste** dataset. All ~4,800 images were captured directly from a municipal waste landfill facility's sorting conveyor belt, training the model on authentic, crumpled, dirty, and crushed waste.

### 2. The Color-Channel Mismatch (BGR vs. RGB)
* **The Issue:** OpenCV captures webcam frames in the **BGR** (Blue, Green, Red) color space. However, deep learning vision models (such as ResNet18 pre-trained on ImageNet) expect **RGB** (Red, Green, Blue) images. Feeding BGR arrays directly causes color channels to swap, leading to incorrect classifications (e.g., green vegetation classified as cardboard due to blue-red color inversion).
* **The Solution:** We implement explicit BGR-to-RGB color space conversion (`cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)`) before converting arrays to PIL Images for normal PyTorch tensor transformations.

### 3. The Video Frame Stutter (UI Blocking)
* **The Issue:** Running PyTorch model inference on a standard CPU takes approximately 100–200ms per frame. If executed sequentially in the main camera feed loop, the display frame rate drops to 4–5 FPS, causing heavy UI lag and stutter.
* **The Solution:** EcoSort AI runs inference inside a **background daemon thread** (`inference_worker`). The main thread pushes video frames to a shared thread-safe buffer, and the worker thread continuously performs predictions. This isolates the CPU latency, allowing the camera window to display at a smooth, buttery **30+ FPS**.

### 4. The Prediction Jitter (Frame Noise)
* **The Issue:** Single-frame video classifications can flicker or fluctuate between adjacent classes (e.g., classifying a metal can as plastic for a split second due to lighting glare).
* **The Solution:** We implement **Temporal Prediction Smoothing** using a rolling history queue (`collections.deque` of size 10). The dashboard displays the majority predicted class across the last 10 frames, creating stable, noise-free, and highly accurate detections.

---

## 🌟 Key Features

* **Futuristic Cyberpunk HUD:** A dark-themed 1280x720 console showing telemetry, confidence bars, and local sorting instructions.
* **Dynamic Window Maximization:** Supports aspect-ratio-preserving scaling. Maximizing or resizing the window keeps the camera viewport, neon Scanning Zone, text labels, and sidebar perfectly centered and aligned.
* **Waste Scanning Zone (ROI):** Neon brackets in the center of the camera feed isolate the object from background clutter (walls, faces, clothes), feeding only the relevant crop to the model.
* **Interactive Guidance Hints:** Overlays real-time feedback (e.g., `PLACE ITEM STABLE INSIDE SCAN BOX`, `HINT: HOLD ITEM CLOSER & STEADY`, or `SCAN SUCCESSFUL! ROUTE TO BIN.`) depending on confidence thresholds.
* **Live Session Metrics:** Computes scan count metrics, recycler and compost tallies, and landfill diversion rates on the fly.
* **Smart Double-Count Prevention:** Telemetry counters only increment when a new class stabilizes, resetting itself only when the user clears the Scanning Zone.

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.10+** installed on your system.

### 1. Clone the Repository
```bash
git clone <repository_url>
cd EcoSort-AI
```

### 2. Configure Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Training Pipeline (Optional)
If you want to re-train or fine-tune the ResNet18 model on the RealWaste dataset, ensure the `RealWaste/` folder exists and run:
```powershell
python train.py
```
*This script fixes subset splitting transforms, applies `ColorJitter` data augmentations to adapt to webcam lighting, trains for 5 epochs on CPU/GPU, and saves weights to `best_waste_model.pth`.*

### 4. Launch the Webcam Classifier Console
```powershell
python live_detector.py
```

---

## ⌨️ Console Controls

* **`q`** or **`ESC`**: Quit the application gracefully.
* **`r`**: Reset session statistics and telemetry counters.

---

## 📁 Codebase Directory Map

* **`live_detector.py`**: Main desktop application. Contains the OpenCV GUI loop, background threading, temporal smoothing state machine, and HUD rendering logic.
* **`train.py`**: Model training script. Implements the custom `SubsetWrapper` for clean data augmentation splits.
* **`app.py`**: Fallback Streamlit-based web console for manual static image uploads.
* **`requirements.txt`**: Declares required Python libraries (`torch`, `torchvision`, `opencv-python`, `streamlit`, `matplotlib`, `pandas`, `pillow`).
* **`best_waste_model.pth`**: Trained model weights (automatically loaded on startup).
* **`dataset_analyzer.py`**: Helper script to scan and check the integrity of dataset files.

---

## ♻️ Waste Categorization Map

The model classifies waste into 9 authentic categories from the landfill conveyor line:

| Category | Recyclable? | Target Bin | Color Code |
| :--- | :---: | :--- | :--- |
| **Cardboard** | Yes | Recycle Bin (Blue) | Blue |
| **Food Organics** | Yes | Compost Bin (Green) | Green |
| **Glass** | Yes | Glass Bin / Recycle | Yellow |
| **Metal** | Yes | Recycle Bin (Blue) | Blue |
| **Miscellaneous Trash** | No | Landfill Bin (Black) | Dark Gray |
| **Paper** | Yes | Recycle Bin (Blue) | Blue |
| **Plastic** | Yes | Recycle Bin (Blue) | Blue |
| **Textile Trash** | No | Fabric Recycling / Donation | Purple |
| **Vegetation** | Yes | Yard Waste Bin (Green) | Green |