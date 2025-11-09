---
title: Derma Scanner
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
python_version: "3.10"
license: mit
pinned: false
---

# Derma Scanner
HAM10000-based demo. **Not** a medical device; **not** a diagnosis.

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
- **Docker** (optional, for containerized deployment)

## Architecture Diagram
```
flowchart LR
    A[Browser / Client] -- upload image --> B[Gradio UI]
    A -- POST /predict --> C[FastAPI]
    B -- local call --> D[Predictor (PyTorch)]
    C -- local call --> D[Predictor (PyTorch)]
    D -->|top-1 label + probabilities| B
    D -->|JSON| C
    subgraph Server
      B
      C
      D
    end
```
## How does it work
1. At startup, the app loads a PyTorch model (`models/derma_scanner_efficientnet_b0.pth`).
2. When an image is uploaded, it is resized and normalized to the model’s expected input.
3. The CNN outputs logits → softmax probabilities → top‑k predictions.
4. The UI shows the predicted class and a probability bar chart.

## Development process
1. **Get the dataset**  
   - Download HAM10000 images and metadata (`HAM10000_metadata.csv`).  
   - Put everything under `data/raw/`:

   ```text
   data/raw/
     HAM10000_images_part_1/
     HAM10000_images_part_2/
     HAM10000_metadata.csv
   ```

2. **Prepare folders**  
   - Create stratified **train/val/test** splits by running:
   ```bash
   python src/data_prepare.py
   ```
   This will create:
   ```text
   data/processed/
     train/<class>/*.jpg
     val/<class>/*.jpg
     test/<class>/*.jpg
   ```

3. **Train the model**  
   ```bash
   python src/train.py --epochs 8 --batch-size 32 --lr 3e-4 --model efficientnet_b0
   ```
   This saves `models/derma_scanner_efficientnet_b0.pth` and `src/labels.json`.

4. **Run the web app locally**  
   ```bash
   # Start the Gradio+FastAPI hybrid app
   python app.py
   # Gradio UI prints a local URL (e.g., http://127.0.0.1:7860)
   # FastAPI runs at http://127.0.0.1:8000 (or as printed)
   ```

5. **Deploy a live app**
   - **Hugging Face Spaces (Gradio):**
     - Create a Space → type **Gradio** → upload this repo, ensure `requirements.txt` and `app.py` are present.  
     - Set `app.py` as the entry point.
   - **Render / Railway (FastAPI):**
     - Use `uvicorn app:api --host 0.0.0.0 --port 8000` as the Start Command.
     - Add a **Background Worker** or a separate service for Gradio if desired, or keep only FastAPI.
   - **Docker (any cloud):**
     ```bash
     docker build -t derma-scanner .
     docker run -p 7860:7860 -p 8000:8000 derma-scanner
     ```

## Screenshots
Add screenshots of the UI under `assets/` and reference them here in the README.

## Demo
- Local Gradio: prints a URL when you run `python app.py`  
- Deployed demo: add your Hugging Face Spaces URL here after deployment.

## Team
- Add names, roles, and responsibilities here.

---

### Quickstart (everything)
```bash
# 0) (optional) Create venv
python -m venv .venv && source .venv/bin/activate

# 1) Install deps
pip install -r requirements.txt

# 2) Prepare data
python src/data_prepare.py

# 3) Train
python src/train.py --epochs 8

# 4) Launch app
python app.py
```
