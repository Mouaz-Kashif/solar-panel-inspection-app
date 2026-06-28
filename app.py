import base64
import io
import zipfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

# =========================
# App config
# =========================
st.set_page_config(page_title="Solar Panel Inspection", page_icon="☀️", layout="wide")

# =========================
# Style (futuristic console)
# =========================
st.markdown(
    """
<style>
:root {
  --bg0: #05070B;
  --bg1: #09131E;
  --stroke: rgba(255,255,255,0.11);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.66);
  --solar: #FFD166;   /* sunlight */
  --ion:   #4CC9F0;   /* electricity */
  --vio:   #7C3AED;   /* futuristic accent */
}

.stApp {
  background:
    radial-gradient(1200px 600px at 20% 0%, rgba(255,209,102,0.10), transparent 55%),
    radial-gradient(900px 500px at 90% 10%, rgba(76,201,240,0.10), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

.block-container {padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 1200px;}

h1, h2, h3, h4 {letter-spacing: -0.02em;}

section[data-testid="stSidebar"] {
  background:
    radial-gradient(900px 600px at 30% 10%, rgba(124,58,237,0.10), transparent 55%),
    radial-gradient(1000px 700px at 70% 30%, rgba(76,201,240,0.10), transparent 60%),
    linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.55));
  border-right: 1px solid var(--stroke);
}

.card {
  position: relative;
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.40);
}

.card:before {
  content: "";
  position: absolute;
  left: 14px;
  right: 14px;
  top: 10px;
  height: 2px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255,209,102,0.0), rgba(255,209,102,0.55), rgba(76,201,240,0.55), rgba(124,58,237,0.0));
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
.bigtitle {font-size: 2.35rem; font-weight: 850; color: var(--text); margin: 0.15rem 0 0.35rem 0;}
.sub {color: var(--muted); margin: 0; max-width: 75ch; line-height: 1.45;}
.badge {
  display:inline-block; padding: 0.35rem 0.65rem; border-radius: 999px;
  background: rgba(76,201,240,0.14); border: 1px solid rgba(76,201,240,0.28);
  color: var(--text); font-size: 0.82rem;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--stroke);
  border-radius: 14px;
  overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Constants
# =========================
CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]

MODEL_URL = "https://github.com/Mouaz-Kashif/solar-panel-inspection-app/releases/download/v1.0-model/best_efficientnetv2.pt"

ASSET_BASE = "https://github.com/Mouaz-Kashif/solar-panel-inspection-app/releases/download/v1.0-assets"
ASSETS = {
    "confusion_matrix_best": f"{ASSET_BASE}/confusion_matrix_efficientnetv2.pdf",
    "train_curves_best": f"{ASSET_BASE}/train_val_curves_efficientnetv2.pdf",
    "model_comparison_pdf": f"{ASSET_BASE}/model_comparison.pdf",
    "model_comparison_csv": f"{ASSET_BASE}/model_comparison.csv",
    "gradcam_pdf": f"{ASSET_BASE}/gradcam_examples_best_model.pdf",
    "ig_pdf": f"{ASSET_BASE}/integrated_gradients_best_model.pdf",
    "demo_zip": f"{ASSET_BASE}/demo_images.zip",
}

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
def download_bytes(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


@st.cache_resource(show_spinner=False)
def load_model() -> nn.Module:
    weights = EfficientNet_V2_S_Weights.DEFAULT
    m = efficientnet_v2_s(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, len(CLASS_NAMES))

    b = download_bytes(MODEL_URL, timeout=180)
    state = torch.load(io.BytesIO(b), map_location="cpu")
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


# =========================
# Demo images (download + index)
# =========================
@st.cache_resource(show_spinner=False)
def get_demo_images_index():
    demo_root = Path(".demo_images")
    if not demo_root.exists():
        demo_root.mkdir(parents=True, exist_ok=True)
        zbytes = download_bytes(ASSETS["demo_zip"], timeout=180)
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            zf.extractall(demo_root)

    base = demo_root / "demo_images"
    idx = {c: [] for c in CLASS_NAMES}
    if base.exists():
        for c in CLASS_NAMES:
            p = base / c
            if p.exists():
                idx[c] = sorted([x for x in p.iterdir() if x.is_file()])
    return idx


def load_random_demo_pil(class_name: str) -> Image.Image | None:
    import random

    files = get_demo_images_index().get(class_name, [])
    if not files:
        return None
    return Image.open(random.choice(files)).convert("RGB")


# =========================
# Session state (persist image across reruns, e.g., slider)
# =========================
if "pil_bytes" not in st.session_state:
    st.session_state["pil_bytes"] = None
if "pil_name" not in st.session_state:
    st.session_state["pil_name"] = None


def set_pil(pil: Image.Image, name: str):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    st.session_state["pil_bytes"] = buf.getvalue()
    st.session_state["pil_name"] = name


def get_pil() -> Image.Image | None:
    b = st.session_state.get("pil_bytes")
    if not b:
        return None
    return Image.open(io.BytesIO(b)).convert("RGB")


# =========================
# Navigation
# =========================
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


@st.cache_resource(show_spinner=False)
def get_model_and_layer():
    m = load_model()
    layer = get_target_layer(m)
    return m, layer


# =========================
# Page: Live demo
# =========================
if page == "Live demo":
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Explainable computer vision for renewable-energy maintenance</div>
  <div class="bigtitle">Solar Panel Dust & Fault Classification</div>
  <p class="sub">Upload a solar panel image or use a dataset sample. Adjust heatmap strength without losing your input.</p>
  <span class="badge">Best model: EfficientNetV2 (transfer learning)</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Input")
        st.caption("Use an example from the dataset or upload your own image.")

        bcols = st.columns(3)
        sample_pick = None
        for i, cname in enumerate(CLASS_NAMES):
            with bcols[i % 3]:
                if st.button(f"Try: {cname}", use_container_width=True):
                    sample_pick = cname

        if sample_pick is not None:
            with st.spinner("Loading sample image..."):
                pil_sample = load_random_demo_pil(sample_pick)
                if pil_sample is None:
                    st.warning("No demo images found for that class. Check demo_images.zip structure in v1.0-assets.")
                else:
                    set_pil(pil_sample, f"sample_{sample_pick}.png")

        up = st.file_uploader(
            "Solar panel photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if up is not None:
            try:
                pil_up = Image.open(up).convert("RGB")
                set_pil(pil_up, up.name)
            except Exception:
                st.error("Could not read that file as an image.")

        pil = get_pil()
        if pil is None:
            st.info("Upload an image or click a sample button.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.image(pil, caption=st.session_state.get("pil_name") or "Input image", use_container_width=True)
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

        heat_alpha = st.slider("Heatmap strength", min_value=0.15, max_value=0.75, value=0.45, step=0.05)

        with st.spinner("Generating Grad‑CAM..."):
            cam, _ = grad_cam(model, layer, x, class_idx=pred_idx)
            overlay = overlay_heatmap(pil, cam, alpha=heat_alpha)

        st.image(overlay, caption="Grad‑CAM overlay (what the model focused on)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Prototype decision-support tool. Verify predictions with a technician before maintenance actions.")


# =========================
# Page: Project information
# =========================
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


# =========================
# Page: Model performance
# =========================
elif page == "Model performance":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Held‑out test set</div>", unsafe_allow_html=True)
    st.markdown("<div class='bigtitle'>Model performance</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>Test‑set evaluation dashboard. Values come from the Kaggle notebook run and are served here as release assets.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    import pandas as pd

    with st.spinner("Loading evaluation assets..."):
        dfc = pd.read_csv(io.BytesIO(download_bytes(ASSETS["model_comparison_csv"], timeout=120)))

    best_row = dfc.sort_values("test_macro_f1", ascending=False).iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model", str(best_row["model"]))
    m2.metric("Macro‑F1", f"{float(best_row['test_macro_f1']):.3f}")
    m3.metric("Accuracy", f"{float(best_row['test_accuracy']):.3f}")
    m4.metric("Params", f"{int(best_row['params']):,}")

    st.write("")
    st.subheader("Comparison table")
    show = dfc.copy()
    for col in ["test_accuracy", "test_macro_f1", "inference_time_sec_11_batches"]:
        if col in show.columns:
            show[col] = show[col].astype(float).map(lambda x: f"{x:.4f}")
    if "params" in show.columns:
        show["params"] = show["params"].astype(int).map(lambda x: f"{x:,}")

    st.dataframe(show, use_container_width=True, hide_index=True)

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")

    with c1:
        st.subheader("Macro‑F1 figure")
        b = download_bytes(ASSETS["model_comparison_pdf"], timeout=120)
        st.download_button("Download model_comparison.pdf", data=b, file_name="model_comparison.pdf", mime="application/pdf")

    with c2:
        st.subheader("Best model confusion matrix")
        b = download_bytes(ASSETS["confusion_matrix_best"], timeout=120)
        st.download_button("Download confusion_matrix_efficientnetv2.pdf", data=b, file_name="confusion_matrix_efficientnetv2.pdf", mime="application/pdf")

    with c3:
        st.subheader("Training curves")
        b = download_bytes(ASSETS["train_curves_best"], timeout=120)
        st.download_button("Download train_val_curves_efficientnetv2.pdf", data=b, file_name="train_val_curves_efficientnetv2.pdf", mime="application/pdf")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page: Why trust the model?
# =========================
elif page == "Why trust the model?":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div class='bigtitle'>Why trust the model?</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>High accuracy alone is not enough. We verify that the model attends to panel-surface evidence rather than background shortcuts.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    def embed_pdf_from_url(url: str, title: str, height: int = 640):
        b = download_bytes(url, timeout=120)
        st.download_button(
            f"Download {title}.pdf",
            data=b,
            file_name=f"{title}.pdf",
            mime="application/pdf",
            use_container_width=False,
        )

        st.info(
            "Browser security can block embedded PDFs in iframes on Streamlit. "
            "Use the download button above to view the PDF, or open it in a new tab."
        )

        # Try embedding anyway (works in some browsers), but don't rely on it.
        b64 = base64.b64encode(b).decode("utf-8")
        html = f"""<iframe sandbox="allow-same-origin allow-scripts" src="data:application/pdf;base64,{b64}" width="100%" height="{height}" style="border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;"></iframe>"""
        st.markdown(html, unsafe_allow_html=True)

    st.subheader("Grad‑CAM (best model)")
    embed_pdf_from_url(ASSETS["gradcam_pdf"], "gradcam_examples_best_model")

    st.write("")
    st.subheader("Integrated Gradients (best model)")
    embed_pdf_from_url(ASSETS["ig_pdf"], "integrated_gradients_best_model")

    st.caption("Figures exported from Kaggle as PDFs and served here as release assets.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page: About
# =========================
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
    st.write("• Email: mouaz.kashif@example.com")
    st.write("• LinkedIn: https://www.linkedin.com/in/mouaz-kashif")
    st.write("• GitHub: https://github.com/Mouaz-Kashif")
    st.write("• Project: https://solar-panel-inspection-app.streamlit.app/")
    st.markdown("</div>", unsafe_allow_html=True)
