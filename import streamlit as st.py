import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# STEP 1: LAYOUT CONFIGURATION
# ==========================================
st.set_page_config(page_title="HSV Threshold Explorer", layout="wide")
st.title("🎛️ Interactive HSV Threshold Explorer")
st.write("Upload an image and adjust the sliders to isolate specific color regions using computer vision masking.")

# ==========================================
# STEP 2: SIDEBAR CONTROL CONTROLS
# ==========================================
st.sidebar.header("🎨 HSV Range Calibration")
st.sidebar.write("OpenCV maps Hue from 0-180, and Saturation/Value from 0-255.")

# Preset suggestions to give users a starting baseline
preset = st.sidebar.selectbox(
    "Load a Color Template Base:",
    ("Manual Control", "Green Range", "Blue Range", "Yellow Range", "Red Range")
)

# Set defaults based on the chosen template baseline
defaults = {"h_min": 0, "h_max": 180, "s_min": 40, "s_max": 255, "v_min": 40, "v_max": 255}
if preset == "Green Range":
    defaults.update({"h_min": 35, "h_max": 85})
elif preset == "Blue Range":
    defaults.update({"h_min": 100, "h_max": 140})
elif preset == "Yellow Range":
    defaults.update({"h_min": 20, "h_max": 35})
elif preset == "Red Range":
    defaults.update({"h_min": 0, "h_max": 15}) # Note: Red also wraps around 170-180

# Sliders for Hue bounds
h_min, h_max = st.sidebar.slider(
    "Hue Bounds (Color Type)", 
    0, 180, (defaults["h_min"], defaults["h_max"])
)

# Sliders for Saturation bounds
s_min, s_max = st.sidebar.slider(
    "Saturation Bounds (Vibrancy)", 
    0, 255, (defaults["s_min"], defaults["s_max"])
)

# Sliders for Value bounds
v_min, v_max = st.sidebar.slider(
    "Value Bounds (Brightness)", 
    0, 255, (defaults["v_min"], defaults["v_max"])
)

# Assemble bounds arrays
lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

# ==========================================
# STEP 3: IMAGE UPLOAD & HANDLING
# ==========================================
uploaded_file = st.file_uploader("Choose an image to isolate...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image and make sure it's standard 8-bit RGB format
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_image)
    
    # OpenCV expects BGR formatting for its color spaces transformations
    bgr_image = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    # ==========================================
    # STEP 4: MASK GENERATION PIPELINE
    # ==========================================
    # Generate the binary mask (inRange yields 255 for match, 0 for skip)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Isolate color by multiplying the binary mask back into original RGB arrays
    isolated_output = cv2.bitwise_and(original_np, original_np, mask=mask)
    
    # ==========================================
    # STEP 5: DASHBOARD LAYOUT PRESENTATION
    # ==========================================
    st.markdown("---")
    
    # Break UI into three uniform horizontal columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📸 1. Input Image")
        st.image(original_np, use_container_width=True)
        
    with col2:
        st.subheader("🎭 2. Binary Mask View")
        # Displaying the single-channel mask layout directly
        st.image(mask, use_container_width=True, channels="GRAY")
        
    with col3:
        st.subheader("🎯 3. Segmented Output")
        st.image(isolated_output, use_container_width=True)
        
    # Informational readout metrics block
    st.info(
        f"**Active Threshold Settings:** "
        f"Lower HSV: `[{lower_bound[0]}, {lower_bound[1]}, {lower_bound[2]}]` | "
        f"Upper HSV: `[{upper_bound[0]}, {upper_bound[1]}, {upper_bound[2]}]`"
    )
else:
    # ==========================================
    # STEP 6: PLACEHOLDER STATE (NO IMAGE YET)
    # ==========================================
    st.info("💡 **Getting Started:** Drop a JPG or PNG file into the uploader above to begin color segmenting!")
    
    # Quick visual cheat-sheet for the user
    with st.expander("ℹ️ Quick HSV Primer"):
        st.markdown("""
        - **Hue (0–180):** Represents the pure color type (e.g., Red, Orange, Green).
        - **Saturation (0–255):** Represents the intensity or vibrancy of the color (0 is gray, 255 is fully saturated).
        - **Value (0–255):** Represents the brightness of the color (0 is pitch black, 255 is full brightness).
        """)