import io
import os
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcoSort-Backend")

app = FastAPI(title="EcoSort AI Waste Classification Service")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "../best_waste_model.pth"
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

# Waste sorting rules mapping
WASTE_METADATA = {
    "cardboard": {
        "title": "Cardboard",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "instructions": [
            "Flatten boxes completely to save space.",
            "Remove all packaging tape, plastic straps, and bubble wraps.",
            "Must be clean and dry. Throw greasy pizza boxes in the compost/trash."
        ],
        "impact": "Recycling 1 ton of cardboard saves 46 gallons of oil and 9 cubic yards of landfill space."
    },
    "food organics": {
        "title": "Food Organics",
        "recyclable": True,
        "bin": "Compost Bin (Green)",
        "instructions": [
            "Place fruit skins, vegetable scraps, leftover meals, and tea bags here.",
            "Ensure no plastic stickers, plastic wraps, or metal twist ties are attached.",
            "Paper towels soiled with food are also compostable."
        ],
        "impact": "Composting redirects organic waste to create nutrient-rich soil and reduces landfill gas."
    },
    "glass": {
        "title": "Glass Bottles & Jars",
        "recyclable": True,
        "bin": "Glass Bin / Recycle Bin (Blue)",
        "instructions": [
            "Empty and rinse all glass containers.",
            "Remove metal caps and rings (metal can be recycled separately).",
            "Do not mix drinking glasses, Pyrex, window glass, or mirrors (non-recyclable)."
        ],
        "impact": "Glass is 100% recyclable and can be recycled endlessly without loss of quality."
    },
    "metal": {
        "title": "Metal & Cans",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "instructions": [
            "Rinse out aluminum soda cans and steel soup/food cans.",
            "Crumple clean aluminum foil into a ball (fist-sized) before recycling.",
            "No need to remove paper labels from tin/steel cans."
        ],
        "impact": "Making cans from recycled aluminum uses 95% less energy than using raw bauxite."
    },
    "miscellaneous trash": {
        "title": "General Trash / Landfill",
        "recyclable": False,
        "bin": "Landfill Bin (Black/Gray)",
        "instructions": [
            "For non-recyclable materials like diapers, chip bags, wipes, and Styrofoam.",
            "Make sure items are dry and bag them securely.",
            "Try to substitute single-use items in this category with reusable alternatives."
        ],
        "impact": "Landfill waste stays buried for centuries. Minimizing this bin is our primary goal."
    },
    "paper": {
        "title": "Mixed Paper",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "instructions": [
            "Includes clean mail, notebooks, flyers, envelopes, and paper bags.",
            "Shredded paper must be placed in a paper bag to prevent it blowing away during collection.",
            "Ensure paper is clean and dry. Wet paper cannot be recycled."
        ],
        "impact": "Recycling one ton of paper saves 17 trees, 7,000 gallons of water, and 3.3 cubic yards of landfill."
    },
    "plastic": {
        "title": "Recyclable Plastics",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "instructions": [
            "Includes plastic bottles, juice jugs, detergent containers, and tubs (types #1 and #2).",
            "Rinse all residue thoroughly. Crush bottles to save bin space.",
            "Soft plastics (grocery bags, bubble wrap) must be returned to store drop-offs, not curbside."
        ],
        "impact": "Plastic takes up to 500 years to decompose. Recycling prevents plastic ocean pollution."
    },
    "textile trash": {
        "title": "Textile Trash / Clothing",
        "recyclable": False,
        "bin": "Donation / Fabric Recycling",
        "instructions": [
            "Donate wearable clothing and shoes to charity or donation bins.",
            "Unwearable textiles can be cut up into cleaning rags.",
            "Look for specialized municipal fabric collection centers."
        ],
        "impact": "Nearly 85% of textiles end up in landfills. Repurposing fabrics prevents landfill clogging."
    },
    "vegetation": {
        "title": "Yard Waste / Vegetation",
        "recyclable": True,
        "bin": "Yard Waste Bin (Green)",
        "instructions": [
            "Includes grass clippings, leaves, twigs, weeds, and small branches.",
            "Do not mix rocks, soil, plastic bags, or treated wood here.",
            "Place loose in the bin or use approved paper yard bags."
        ],
        "impact": "Yard waste is composted on a large scale to provide high-quality mulch for farms and parks."
    }
}

# Image transform pipeline (must match val_transform in train.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global model state
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    global model
    logger.info("Checking for trained model weights at %s...", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try:
            logger.info("Trained model found. Initializing ResNet18...")
            # Recreate model architecture
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            
            # Load weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()
            logger.info("Trained model loaded successfully!")
        except Exception as e:
            logger.error("Failed to load local model weights: %s", str(e))
            model = None
    else:
        logger.warning("No local trained model found yet. Server will run in Sandbox Mode.")

@app.get("/api/health")
def health_check():
    local_model_loaded = model is not None
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": local_model_loaded,
        "message": "EcoSort API Online using trained model." if local_model_loaded else "EcoSort API Online in Sandbox Mode (inference will be simulated)."
    }

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    global model
    start_time = time.time()

    # Verify and parse image file
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error("Failed to parse image file: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # 1. Model inference if trained model is loaded
    if model is not None:
        try:
            # Prepare image
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, dim=0)
            
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
            predicted_class = CLASS_NAMES[predicted_idx]
            
            meta_key = predicted_class.lower()
            meta = WASTE_METADATA.get(meta_key, WASTE_METADATA["miscellaneous trash"])
            latency = time.time() - start_time

            return {
                "prediction": meta_key,
                "confidence": round(confidence, 4),
                "label": meta["title"],
                "recyclable": meta["recyclable"],
                "bin": meta["bin"],
                "instructions": meta["instructions"],
                "impact": meta["impact"],
                "latency_seconds": round(latency, 3),
                "is_mocked": False
            }
        except Exception as e:
            logger.error("Inference failure, falling back: %s", str(e))

    # 2. Simulated Sandbox Fallback
    logger.info("Simulating classification result...")
    time.sleep(0.5) # small delay to feel like CPU inference
    
    filename = file.filename.lower()
    if "plastic" in filename or "bottle" in filename:
        predicted_class = "plastic"
    elif "can" in filename or "metal" in filename or "tin" in filename:
        predicted_class = "metal"
    elif "paper" in filename or "news" in filename:
        predicted_class = "paper"
    elif "card" in filename or "box" in filename:
        predicted_class = "cardboard"
    elif "glass" in filename or "jar" in filename:
        predicted_class = "glass"
    elif "food" in filename or "apple" in filename or "banana" in filename:
        predicted_class = "food organics"
    elif "leave" in filename or "grass" in filename or "branch" in filename:
        predicted_class = "vegetation"
    elif "cloth" in filename or "shirt" in filename or "shoe" in filename:
        predicted_class = "textile trash"
    else:
        # Pick plastic or random as a default
        import random
        predicted_class = random.choice(list(WASTE_METADATA.keys()))

    meta = WASTE_METADATA.get(predicted_class, WASTE_METADATA["miscellaneous trash"])
    confidence = 0.82 + (hash(file.filename) % 17) / 100.0  # mock confidence between 82% and 98%
    latency = time.time() - start_time

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "label": meta["title"],
        "recyclable": meta["recyclable"],
        "bin": meta["bin"],
        "instructions": meta["instructions"],
        "impact": meta["impact"],
        "latency_seconds": round(latency, 3),
        "is_mocked": True
    }
