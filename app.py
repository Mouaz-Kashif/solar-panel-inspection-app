import io
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Solar Panel Inspection",
    page_icon="☀️",
    layout="wide",
)

# -------------------------
# Style (opinionated, distinct)
# -------------------------
st.markdown(
    """
<style>
:root {
  --bg0: #060A0F;
  --bg1: #0A1420;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --solar: #FFD166;   /* warm sunlight */
  --ion:   #4CC9F0;   /* cool electric */
  --warn:  #F25F5C;
}

.block-container {padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 1200px;}

h1, h2, h3, h4 {letter-spacing: -0.02em;}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: radial-gradient(1200px 800px at 20% 10%, rgba(76,201,240,0.10), transparent 60%),
              radial-gradient(1000px 700px at 70% 30%, rgba(255,209,102,0.10), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
  border-right: 1px solid var(--stroke);
}

/* Cards */
.card {
  background: linear-gradient(180deg, var(--card), rgba(255,255,255,0.03));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}

.hero {
  border-radius: 24px;
  padding: 22px 22px;
  background:
    radial-gradient(900px 420px at 10% 10%, rgba(255,209,102,0.22), transparent 60%),
    radial-gradient(800px 380px at 90% 20%, rgba(76,201,240,0.20), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border: 1px solid var(--stroke);
}

.kicker {color: var(--muted); text-transform: uppercase; letter-spacing: 0.18em; font-size: 0.75rem;}
.bigtitle {font-size: 2.25rem; font-weight: 800; color: var(--text); margin: 0.15rem 0 0.35rem 0;}
.sub {color: var(--muted); margin: 0; max-width: 70ch;}
.badge {
  display:inline-block; padding: 0.35rem 0.6rem; border-radius: 999px;
  background: rgba(255,209,102,0.15); border: 1px solid rgba(255,209,102,0.30);
  color: var(--text); font-size: 0.80rem;
}
.small {color: var(--muted); font-size: 0.9rem;}

/* Tables */
[data-testid="stTable"] {border: 1px solid var(--stroke); border-radius: 14px; overflow: hidden;}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Constants
# -------------------------
CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]

MODEL_URL = "https://github.com/Mouaz-Kashif/solar-panel-inspection-app/releases/download/v1.0-model/best_efficientnetv2.pt"
MODEL_LOCAL = Path("best_efficientnetv2.pt")

IMG_SIZE = 224

VAL_TFMS = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data(show_spinner=False)
def download_model_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


@st.cache_resource(show_spinner=False)
def load_model() -> nn.Module:
    weights = EfficientNet_V2_S_Weights.DEFAULT
    m = efficientnet_v2_s(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, len(CLASS_NAMES))

    if MODEL_LOCAL.exists():
        state = torch.load(MODEL_LOCAL, map_location="cpu")
    else:
        b = download_model_bytes(MODEL_URL)
        state = torch.load(io.BytesIO(b), map_location="cpu")

    # Be tolerant to minor version differences
    m.load_state_dict(state, strict=True)
    m.eval()
    m.to(device())
    return m


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    x = VAL_TFMS(pil.convert("RGB")).unsqueeze(0)
    return x.to(device())


@torch.no_grad()
def predict(m: nn.Module, x: torch.Tensor):
    logits = m(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return idx, probs


def get_target_layer(m: nn.Module):
    return m.features[-1]


def grad_cam(m: nn.Module, target_layer: nn.Module, x: torch.Tensor, class_idx: int | None = None):
    m.eval()

    activations = None
    gradients = None

    def fwd_hook(_module, _inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(_module, _grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = m(x)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())

    m.zero_grad(set_to_none=True)
    score = logits[:, class_idx].sum()
    score.backward(retain_graph=True)

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()

    return cam, class_idx


def overlay_heatmap(pil: Image.Image, heat: np.ndarray, alpha: float = 0.45):
    import matplotlib.cm as cm

    img = np.array(pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    heat_t = torch.tensor(heat)[None, None, ...]
    heat_t = F.interpolate(heat_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    heat = heat_t.squeeze().numpy()

    heat_rgb = cm.inferno(heat)[..., :3]
    over = (1 - alpha) * img + alpha * heat_rgb
    return np.clip(over, 0, 1)


# -------------------------
# Navigation
# -------------------------
PAGES = [
    "Live demo",
    "Project information",
    "Model performance",
    "Why trust the model?",
    "About",
]

with st.sidebar:
    st.markdown("<div class='kicker'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", PAGES, index=0)
    st.divider()

    st.markdown("<div class='kicker'>System</div>", unsafe_allow_html=True)
    st.write("**Backbone:** EfficientNetV2-S")
    st.write("**Classes:**")
    st.write(", ".join(CLASS_NAMES))


# -------------------------
# Shared model handle
# -------------------------
@st.cache_resource(show_spinner=False)
def get_model_and_layer():
    m = load_model()
    layer = get_target_layer(m)
    return m, layer


# -------------------------
# Page: Live demo
# -------------------------
if page == "Live demo":
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Explainable computer vision for renewable-energy maintenance</div>
  <div class="bigtitle">Solar Panel Dust & Fault Classification</div>
  <p class="sub">Upload a solar panel image to classify its surface condition and generate a Grad‑CAM heatmap showing where the model focused.</p>
  <span class="badge">Best model: EfficientNetV2 (transfer learning)</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Upload image")
        up = st.file_uploader("Solar panel photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

        if up is None:
            st.info("Upload an image to run classification.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        try:
            pil = Image.open(up).convert("RGB")
        except Exception:
            st.error("Could not read that file as an image. Try a JPG/PNG.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.image(pil, caption="Uploaded image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction & explanation")

        with st.spinner("Loading model (first run may take ~10–30s)..."):
            model, layer = get_model_and_layer()

        x = pil_to_tensor(pil)

        with st.spinner("Running inference..."):
            pred_idx, probs = predict(model, x)

        pred_name = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

        a, b = st.columns([1, 1])
        with a:
            st.metric("Predicted class", pred_name)
        with b:
            st.metric("Confidence", f"{conf:.3f}")

        topk = 3
        top_idx = np.argsort(-probs)[:topk]
        st.write("Top‑3 probabilities")
        st.table({"class": [CLASS_NAMES[i] for i in top_idx], "probability": [float(probs[i]) for i in top_idx]})

        st.write("Explanation (Grad‑CAM)")
        heat_alpha = st.slider("Heatmap strength", min_value=0.15, max_value=0.75, value=0.45, step=0.05)

        with st.spinner("Generating Grad‑CAM..."):
            cam, _ = grad_cam(model, layer, x, class_idx=pred_idx)
            overlay = overlay_heatmap(pil, cam, alpha=heat_alpha)

        st.image(overlay, caption="Grad‑CAM overlay (what the model focused on)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Prototype decision-support tool. Verify predictions with a technician before maintenance actions.")


# -------------------------
# Page: Project information
# -------------------------
elif page == "Project information":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>What this project solves</div>", unsafe_allow_html=True)
    st.markdown("<div class='bigtitle'>From manual inspection → explainable visual screening</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>Solar panels lose efficiency due to soiling (dust, bird droppings, snow) and damage (electrical or physical). This project builds an explainable CNN-based classifier to support maintenance prioritization.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset")
        st.write("Kaggle dataset: Solar Panel Images Clean and Faulty Images")
        st.write("Classes: Clean, Dusty, Bird-drop, Snow-Covered, Electrical-damage, Physical-Damage")
        st.write("Split: 70% train / 15% val / 15% test (stratified)")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Approach")
        st.write("• Baseline: Custom CNN")
        st.write("• Transfer learning: EfficientNetV2-S, ConvNeXt-Tiny")
        st.write("• Explainability: Grad-CAM + Integrated Gradients")
        st.write("• Deployment: Streamlit web app")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Page: Model performance
# -------------------------
elif page == "Model performance":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Held‑out test set</div>", unsafe_allow_html=True)
    st.markdown("<div class='bigtitle'>Model comparison</div>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>Paste your final CSV metrics into the app to display a clean comparison dashboard.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload model comparison CSV")
    st.caption("Use the Results/model_comparison.csv generated by your Kaggle notebook.")
    comp_file = st.file_uploader("model_comparison.csv", type=["csv"], key="comp_csv", label_visibility="collapsed")

    if comp_file is None:
        st.info("Upload Results/model_comparison.csv to render the comparison.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    import pandas as pd

    dfc = pd.read_csv(comp_file)
    st.write(dfc)

    # Simple highlights
    best_row = dfc.sort_values("test_macro_f1", ascending=False).iloc[0]
    st.success(f"Best by Macro‑F1: {best_row['model']} (Macro‑F1 = {best_row['test_macro_f1']:.3f}, Acc = {best_row['test_accuracy']:.3f})")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Page: Why trust the model?
# -------------------------
elif page == "Why trust the model?":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div class='bigtitle'>Accuracy isn’t enough</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>We use explanation methods to verify the model focuses on meaningful panel-surface evidence rather than background shortcuts (sky, frames, reflections).</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload explanation figures (PDF) from Kaggle")
    st.caption("Upload your `gradcam_examples_best_model.pdf` and `integrated_gradients_best_model.pdf` to display them here.")

    pdf1 = st.file_uploader("Grad‑CAM PDF", type=["pdf"], key="gc_pdf")
    pdf2 = st.file_uploader("Integrated Gradients PDF", type=["pdf"], key="ig_pdf")

    if pdf1 is None and pdf2 is None:
        st.info("Upload at least one PDF to display explanations.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Streamlit can't render PDFs natively in all browsers, but we can offer download + embed via base64.
    import base64

    def show_pdf(file, title):
        b = file.getvalue()
        st.download_button(f"Download {title}", data=b, file_name=file.name, mime="application/pdf")
        b64 = base64.b64encode(b).decode("utf-8")
        html = f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" style="border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;"></iframe>"""
        st.markdown(html, unsafe_allow_html=True)

    if pdf1 is not None:
        show_pdf(pdf1, "Grad‑CAM")
    if pdf2 is not None:
        show_pdf(pdf2, "Integrated Gradients")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Page: About
# -------------------------
elif page == "About":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='bigtitle'>Mouaz Kashif Shahzad</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>Software Engineering undergraduate building practical, explainable ML systems for real-world infrastructure.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Contact")
    st.write("• Email: your.email@example.com")
    st.write("• LinkedIn: https://www.linkedin.com/in/your-handle")
    st.write("• GitHub: https://github.com/Mouaz-Kashif")
    st.write("• Project: https://solar-panel-inspection-app.streamlit.app/")
    st.markdown("</div>", unsafe_allow_html=True)

