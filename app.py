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
    page_title="Solar Panel Inspection (Dust & Faults)",
    page_icon="☀️",
    layout="wide",
)

st.title("Solar Panel Dust & Fault Classification")
st.caption("Upload a solar panel image to classify surface condition and view an explanation heatmap.")

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

# ImageNet normalization used in training
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
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


@st.cache_resource(show_spinner=False)
def load_model() -> nn.Module:
    # Build architecture (must match training)
    weights = EfficientNet_V2_S_Weights.DEFAULT
    m = efficientnet_v2_s(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, len(CLASS_NAMES))

    # Load weights
    if MODEL_LOCAL.exists():
        state = torch.load(MODEL_LOCAL, map_location="cpu")
    else:
        b = download_model_bytes(MODEL_URL)
        state = torch.load(io.BytesIO(b), map_location="cpu")

    m.load_state_dict(state)
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
    # EfficientNetV2: use last feature block
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

    # activations/grads: [1, C, H, W]
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

    # Resize cam to 224x224
    heat_t = torch.tensor(heat)[None, None, ...]
    heat_t = F.interpolate(heat_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    heat = heat_t.squeeze().numpy()

    heat_rgb = cm.jet(heat)[..., :3]
    over = (1 - alpha) * img + alpha * heat_rgb
    return np.clip(over, 0, 1)


with st.sidebar:
    st.subheader("Model")
    st.write("**Backbone:** EfficientNetV2-S")
    st.write("**Classes:**")
    st.write(", ".join(CLASS_NAMES))
    st.divider()
    st.subheader("Explanation")
    heat_alpha = st.slider("Heatmap strength", min_value=0.15, max_value=0.75, value=0.45, step=0.05)
    st.caption("Tip: higher values make the heatmap more visible.")


col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1) Upload image")
    up = st.file_uploader("Solar panel photo", type=["jpg", "jpeg", "png", "webp"]) 

    if up is None:
        st.info("Upload an image to run classification.")
        st.stop()

    try:
        pil = Image.open(up).convert("RGB")
    except Exception:
        st.error("Could not read that file as an image. Try a JPG/PNG.")
        st.stop()

    st.image(pil, caption="Uploaded image", use_container_width=True)


with col_right:
    st.subheader("2) Prediction + explanation")

    with st.spinner("Loading model (first run may take ~10–30s)..."):
        model = load_model()
        layer = get_target_layer(model)

    x = pil_to_tensor(pil)

    with st.spinner("Running inference..."):
        pred_idx, probs = predict(model, x)

    pred_name = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    st.metric("Predicted class", pred_name, help="Highest-probability class")
    st.metric("Confidence", f"{conf:.3f}")

    topk = 3
    top_idx = np.argsort(-probs)[:topk]
    st.write("Top-3 probabilities")
    st.table(
        {
            "class": [CLASS_NAMES[i] for i in top_idx],
            "probability": [float(probs[i]) for i in top_idx],
        }
    )

    with st.spinner("Generating Grad-CAM heatmap..."):
        cam, _ = grad_cam(model, layer, x, class_idx=pred_idx)
        overlay = overlay_heatmap(pil, cam, alpha=heat_alpha)

    st.image(overlay, caption="Grad-CAM overlay (what the model focused on)", use_container_width=True)

st.divider()
st.caption(
    "Note: This is a prototype decision-support tool. Predictions should be verified by a technician before maintenance actions."
)
