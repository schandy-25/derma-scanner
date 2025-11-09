# src/infer.py
import os
import json
import torch
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Env-configurable (with safe defaults)
HF_WEIGHTS_REPO = os.getenv("HF_WEIGHTS_REPO", "saketti/derma-scanner-weights")
WEIGHTS_FILENAME = os.getenv("WEIGHTS_FILENAME", "derma_scanner_efficientnet_b0.pth")
LABELS_FILENAME  = os.getenv("LABELS_FILENAME", "labels.json")
MODEL_NAME       = os.getenv("MODEL_NAME", "efficientnet_b0")

class Predictor:
    def __init__(self):
        # Download files from HF Hub to local cache (or reuse cached)
        weights_path = hf_hub_download(repo_id=HF_WEIGHTS_REPO, filename=WEIGHTS_FILENAME)
        labels_path  = hf_hub_download(repo_id=HF_WEIGHTS_REPO, filename=LABELS_FILENAME)

        # Load labels
        with open(labels_path, "r") as f:
            self.classes = json.load(f)

        # Device selection (CPU / Apple MPS / CUDA)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build model and load weights
        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval().to(self.device)

        # Transforms
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def predict(self, pil_image: Image.Image):
        x = self.tfm(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy().tolist()
        # top class
        top_idx = int(torch.argmax(logits, dim=1).item())
        return {
            "top_class": self.classes[top_idx],
            "probs": {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        }

# Quick manual test:
if __name__ == "__main__":
    import sys, json as _json
    img = Image.open(sys.argv[1])
    pred = Predictor().predict(img)
    print(_json.dumps(pred, indent=2))
