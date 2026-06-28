# Solar Panel Dust & Fault Classification — Streamlit App

This is a simple web interface for an explainable solar-panel inspection classifier.

## What it does
- Upload a solar panel image
- Predict the surface condition (6 classes)
- Show confidence + top-3 probabilities
- Generate a Grad-CAM heatmap overlay as an explanation

## Model
- Backbone: EfficientNetV2-S (ImageNet pretrained)
- Fine-tuned classifier head for 6 classes
- Weights are downloaded from a GitHub Release asset at runtime

## Classes
- Bird-drop
- Clean
- Dusty
- Electrical-damage
- Physical-Damage
- Snow-Covered

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. On Streamlit Community Cloud, create a new app.
3. Select this repo and set the entry point to `app.py`.
4. Deploy.

## Notes
This is a prototype decision-support tool. Predictions should be verified by a technician before maintenance actions.
