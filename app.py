import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="EcoSort AI - Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode adjustments via Streamlit markup
st.markdown("""
<style>
    .reportview-container {
        background: #0b0f19;
    }
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .bin-box {
        padding: 16px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        margin-bottom: 16px;
        text-align: center;
        font-size: 1.2rem;
    }
    .impact-box {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 4px;
        font-style: italic;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Globals and Metadata
MODEL_PATH = "best_waste_model.pth"
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

WASTE_METADATA = {
    "cardboard": {
        "title": "Cardboard",
        "recyclable": True,
        "bin": "Recycle Bin (Blue)",
        "color": "#3b82f6",
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
        "color": "#10b981",
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
        "color": "#eab308",
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
        "color": "#3b82f6",
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
        "color": "#6b7280",
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
        "color": "#3b82f6",
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
        "color": "#3b82f6",
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
        "color": "#8b5cf6",
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
        "color": "#10b981",
        "instructions": [
            "Includes grass clippings, leaves, twigs, weeds, and small branches.",
            "Do not mix rocks, soil, plastic bags, or treated wood here.",
            "Place loose in the bin or use approved paper yard bags."
        ],
        "impact": "Yard waste is composted on a large scale to provide high-quality mulch for farms and parks."
    }
}

# Image transforms (must match validation transform in train.py)
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Cache Model Loader
@st.cache_resource
def load_resnet_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH):
        try:
            # Load empty architecture
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            # Load trained weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()
            return model, device, "Trained ResNet18 Weights"
        except Exception as e:
            return None, device, f"Failed to load: {e}"
    return None, device, "No trained weights found (Running in Sandbox Mode)"

model, device, status_msg = load_resnet_model()

# 4. Session State Initialization for Stats
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'recyclables' not in st.session_state:
    st.session_state.recyclables = 0
if 'compostables' not in st.session_state:
    st.session_state.compostables = 0

# Sidebar Layout
with st.sidebar:
    st.title("♻️ EcoSort AI Console")
    st.write("---")
    
    # Status display
    st.subheader("System Status")
    if model is not None:
        st.success(f"🟢 Active: {status_msg}")
    else:
        st.warning(f"🟡 Sandbox Mode: {status_msg}")
    st.info(f"Computing on: **{str(device).upper()}**")
    st.write("---")
    
    # Metrics
    st.subheader("Session Statistics")
    st.metric("Total Scans", st.session_state.total_scans)
    st.metric("Recyclables Sorted", st.session_state.recyclables)
    st.metric("Compostables Sorted", st.session_state.compostables)
    
    # Landfill diversion rate
    if st.session_state.total_scans > 0:
        div_rate = int(((st.session_state.recyclables + st.session_state.compostables) / st.session_state.total_scans) * 100)
    else:
        div_rate = 0
    st.metric("Landfill Diversion Rate", f"{div_rate}%")
    
    if st.button("Reset Session Stats"):
        st.session_state.total_scans = 0
        st.session_state.recyclables = 0
        st.session_state.compostables = 0
        st.rerun()

# Main Screen Layout
st.markdown('<div class="main-header">EcoSort AI</div>', unsafe_allow_html=True)
st.write("Point your webcam or upload photos of waste (crumpled, squashed, or dirty) to identify sorting bins and guidelines.")
st.write("---")

col1, col2 = st.columns([1.1, 0.9])

image_input = None

with col1:
    st.subheader("📸 Scanner Input Console")
    
    # Tab navigation
    input_method = st.radio("Choose Input Method:", ["Live Camera Capture", "Upload Image File"], horizontal=True)
    
    if input_method == "Live Camera Capture":
        # Streamlit Camera widget
        camera_photo = st.camera_input("Point camera at your waste item:")
        if camera_photo is not None:
            image_input = Image.open(camera_photo).convert("RGB")
    else:
        # File uploader
        uploaded_file = st.file_uploader("Upload a photo of waste:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file).convert("RGB")
            st.image(image_input, caption="Uploaded Image Preview", use_column_width=True)

# 5. Prediction Execution
prediction_result = None

if image_input is not None:
    with col2:
        st.subheader("🎯 Classification Results")
        
        with st.spinner("Analyzing material textures..."):
            time.sleep(0.3)  # latency simulation feel
            
            # Local ResNet inference
            if model is not None:
                try:
                    tensor = transform_pipeline(image_input).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        confidence, pred_idx = torch.max(probs, dim=0)
                        
                        prediction_key = CLASS_NAMES[pred_idx.item()].lower()
                        confidence_val = confidence.item()
                        is_mocked = False
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    model = None # fallback to mock below
            
            # Sandbox Mock inference
            if model is None:
                is_mocked = True
                confidence_val = 0.88
                # Simulating a default class (Plastic or random)
                import random
                prediction_key = random.choice(list(WASTE_METADATA.keys()))

            # Retrieve Metadata
            meta = WASTE_METADATA.get(prediction_key, WASTE_METADATA["miscellaneous trash"])
            
            # Increment session stats
            st.session_state.total_scans += 1
            if meta["recyclable"]:
                st.session_state.recyclables += 1
            if prediction_key in ["food organics", "vegetation"]:
                st.session_state.compostables += 1

            # Display prediction output
            st.markdown(f"### Predicted Class: **{meta['title']}**")
            st.write(f"**Confidence:** `{confidence_val*100:.2f}%` " + ("(Sandbox Simulation)" if is_mocked else ""))
            
            # Bin Destination header box
            bin_color = meta["color"]
            st.markdown(
                f'<div class="bin-box" style="background-color: {bin_color}">'
                f'ROUTE TO: {meta["bin"]}'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            # Recycle Status Badge
            if meta["recyclable"]:
                st.success("✅ **Recyclable Material**")
            else:
                st.error("⚠️ **Non-Recyclable (Landfill)**")
                
            # Guidelines list
            st.write("**Sorting Directives:**")
            for inst in meta["instructions"]:
                st.write(f"- {inst}")
                
            # Eco Impact block
            st.markdown(
                f'<div class="impact-box">'
                f'💡 **Impact Fact:** {meta["impact"]}'
                f'</div>', 
                unsafe_allow_html=True
            )

else:
    with col2:
        st.subheader("🎯 Classification Results")
        st.info("Awaiting input. Select a photo or take a camera snapshot on the left panel to trigger sorting predictions.")

# 6. Educational panel below
st.write("---")
st.subheader("📊 RealWaste Dataset Profile")
st.write("Traditional trash detectors train on 'perfect', clean products on white backdrops. Our model fine-tunes on the **RealWaste** dataset collected at landfill sorting lines, featuring authentic crumpled, squashed, and degraded materials.")

# Dataset bar chart
chart_data = {
    "Category": ["Vegetation", "Plastic", "Paper", "Food Organics", "Cardboard", "Glass", "Metal", "Miscellaneous Trash", "Textile Trash"],
    "Images Count": [976, 921, 777, 650, 461, 418, 320, 213, 16]
}
df = pd.DataFrame(chart_data)
df = df.set_index("Category")

# Matplotlib styled plot
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

colors = ['#10b981', '#3b82f6', '#3b82f6', '#10b981', '#3b82f6', '#eab308', '#3b82f6', '#6b7280', '#8b5cf6']
bars = ax.barh(df.index[::-1], df["Images Count"][::-1], color=colors[::-1], height=0.6)

# Labels styling
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('#1f2937')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('#1f2937')
ax.spines['right'].set_color('none')
ax.xaxis.grid(True, linestyle='--', alpha=0.1, color='white')
ax.set_axisbelow(True)

st.pyplot(fig)
