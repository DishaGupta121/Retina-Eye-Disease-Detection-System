import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONMALLOC"] = "malloc"

import io
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import cm

# Optional PDF library
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# ===============================
# PAGE CONFIG + CSS
# ===============================
st.set_page_config(page_title="Retina Disease AI Detector", page_icon="👁️", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f1724, #0f4b6d);
    color: #e6eef8;
}
.stApp {
    background: transparent;
}
.card {
    background: rgba(255, 255, 255, 0.04);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 16px;
    backdrop-filter: blur(6px);
}
h1 { color: #ffefc4; }
</style>
""", unsafe_allow_html=True)

# ===============================
# MODEL LOADING
# ===============================
@st.cache_resource
def load_model():
    paths = ["retina_disease_model.keras", "retina_disease_model.h5"]
    for p in paths:
        if os.path.exists(p):
            try:
                return tf.keras.models.load_model(p, compile=False)
            except:
                continue
    st.error("❌ Model file not found.")
    return None

model = load_model()
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# ===============================
# Grad-CAM Utilities (SAFE + FIXED)
# ===============================
def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            continue
    for name in ["Conv_1", "block_16_project", "conv_pw_13_relu"]:
        try:
            model.get_layer(name)
            return name
        except:
            pass
    raise ValueError("No conv layer found")

def generate_gradcam(model, img_array, class_index):
    last_layer_name = find_last_conv_layer_name(model)
    last_conv = model.get_layer(last_layer_name)
    class_output = model.output[0] if isinstance(model.output, (list,tuple)) else model.output

    grad_model = tf.keras.models.Model([model.input], [last_conv.output, class_output])

    with tf.GradientTape() as tape:
        conv_out, prediction = grad_model(img_array)
        loss = prediction[:, class_index]

    grads = tape.gradient(loss, conv_out)
    conv_out = conv_out[0].numpy()
    grads = grads[0].numpy()

    weights = np.mean(grads, axis=(0, 1))
    cam = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_out[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam

def overlay_cam_on_image(img_pil, heatmap, alpha=0.45):
    heatmap = np.uint8(255 * heatmap)
    colored = cm.jet(heatmap / 255.0)[:, :, :3]
    colored_img = Image.fromarray((colored * 255).astype(np.uint8)).resize(img_pil.size)

    return Image.blend(img_pil.convert("RGBA"), colored_img.convert("RGBA"), alpha=alpha)

# ===============================
# PDF HELPER
# ===============================
def pil_to_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()

def create_pdf_report(orig, heat, disease, severity, probs):
    if not HAS_FPDF:
        raise RuntimeError("fpdf missing")

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Retina Disease AI Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 6, f"Predicted Disease: {disease}", ln=True)
    pdf.cell(0, 6, f"Severity Score: {severity:.2f}", ln=True)
    pdf.ln(5)

    for name, p in zip(CLASS_NAMES, probs):
        pdf.cell(0, 6, f"{name}: {p:.3f}", ln=True)
    pdf.ln(5)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    img1 = f"img_{ts}.jpg"
    img2 = f"heat_{ts}.jpg"

    orig.convert("RGB").save(img1)
    heat.convert("RGB").save(img2)

    pdf.image(img1, x=15, w=180)
    pdf.ln(5)
    pdf.image(img2, x=15, w=180)

    with open(img1): pass
    with open(img2): pass

    data = pdf.output(dest="S").encode("latin-1")

    os.remove(img1)
    os.remove(img2)

    return data

# ===============================
# UI
# ===============================
st.markdown("<h1 style='text-align:center;'>👁️ Retina Disease AI Detector</h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("📤 Upload Retina Image", type=["jpg","jpeg","png"])

col1, col2 = st.columns([1,1])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", width=420)
        st.markdown("</div>", unsafe_allow_html=True)

    # preprocess
    arr = img.resize((224,224))
    arr = np.array(arr)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr,0)

    # prediction
    preds = model.predict(arr)

    if len(preds)==2:
        class_pred, reg_pred = preds
        probs = class_pred[0].tolist()
        idx = int(np.argmax(probs))
        disease = CLASS_NAMES[idx]
        severity = float((reg_pred[0][0]+1)/2)
    else:
        probs = preds[0].tolist()
        idx = int(np.argmax(probs))
        disease = CLASS_NAMES[idx]
        severity = float(np.max(probs))

    severity = float(np.clip(severity,0,1))

    # GRAD-CAM
    heatmap = generate_gradcam(model, arr, idx)
    heat_overlay = overlay_cam_on_image(img, heatmap)

    # result
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"### 🧠 Prediction: **{disease}**")
        st.write(f"### 🔥 Severity: **{severity:.2f} / 1.0**")
        st.markdown("---")
        st.write("### Class Probabilities:")
        for cname, p in zip(CLASS_NAMES, probs):
            st.write(f"{cname}: **{p:.3f}**")
            st.progress(p)
        st.markdown("---")
        st.write("### 🔥 Grad-CAM Heatmap")
        st.image(heat_overlay, width=420)
        st.markdown("</div>", unsafe_allow_html=True)

    # PDF
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 📄 Download PDF Report")

    if not HAS_FPDF:
        st.warning("Install FPDF:\n```pip install fpdf```")
    else:
        try:
            orig_small = img.resize((800,800)).convert("RGB")
            heat_small = heat_overlay.resize((800,800)).convert("RGB")

            pdf = create_pdf_report(orig_small, heat_small, disease, severity, probs)

            st.download_button(
                "⬇️ Download Report",
                pdf,
                file_name="retina_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("📸 Upload an image to begin.")
