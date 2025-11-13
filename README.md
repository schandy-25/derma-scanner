
# Derma Scanner
HAM10000-based demo. **Not** a medical device; **not** a diagnosis.
Live app - https://huggingface.co/spaces/saketti/derma-scanner-v2

# Derma Scanner — HAM10000 Skin Lesion Classifier

> ⚠️ **Medical disclaimer:** This demo is for **research and educational purposes only**. It is **not** a medical device and must **not** be used for diagnosis or treatment. Always consult a licensed clinician.

## General Information
**Derma Scanner** is a web app that classifies dermatoscopic images into the 7 HAM10000 categories using a transfer‑learned CNN (EfficientNet‑B0 via `timm`). It provides:
- A Gradio web UI for drag‑and‑drop image classification
- A FastAPI prediction endpoint (`/predict`) for programmatic access
- A reproducible training pipeline with `train.py` and `data_prepare.py`

## Used Dataset
- **HAM10000** (7 classes): `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`  
  Download from Kaggle or the original HAM10000 source. Follow the steps in **Development process → 1. Get the dataset**.

## Used Technologies
- **PyTorch**, **torchvision**, **timm** (EfficientNet‑B0 backbone)
- **Gradio** (simple, robust web UI)
- **FastAPI + Uvicorn** (REST API)
- **scikit‑learn**, **pandas**, **numpy**

```
## How does it work
1. At startup, the app loads a PyTorch model (`models/derma_scanner_efficientnet_b0.pth`).
2. When an image is uploaded, it is resized and normalized to the model’s expected input.
3. The CNN outputs logits → softmax probabilities → top‑k predictions.
4. The UI shows the predicted class and a probability bar chart.

 ```


## Screenshots




https://github.com/user-attachments/assets/cc98ad53-61d2-4f6e-9400-2c0e1e4d1460





