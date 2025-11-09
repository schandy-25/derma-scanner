
import torch, json
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm

class Predictor:
    def __init__(self, weights_path="models/derma_scanner_efficientnet_b0.pth", model_name="efficientnet_b0", labels_path="src/labels.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(labels_path, "r") as f:
            self.classes = json.load(f)
        self.model = timm.create_model(model_name, pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval().to(self.device)
        self.tfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def predict(self, pil_image: Image.Image):
        x = self.tfm(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy().tolist()
        top_idx = int(torch.argmax(logits, dim=1).item())
        return {
            "top_class": self.classes[top_idx],
            "probs": {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        }

# quick manual test
if __name__ == "__main__":
    from PIL import Image
    import sys, json
    img = Image.open(sys.argv[1])
    pred = Predictor().predict(img)
    print(json.dumps(pred, indent=2))
