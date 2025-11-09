
import argparse, os, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tfm_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=tfm_eval)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=tfm_eval)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl, train_ds.classes

def build_model(model_name="efficientnet_b0", num_classes=7):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def evaluate(model, dl, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed", type=str)
    ap.add_argument("--epochs", default=8, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--model", default="efficientnet_b0", type=str)
    ap.add_argument("--img-size", default=224, type=int)
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    train_dl, val_dl, test_dl, classes = get_dataloaders(args.data_dir, args.batch_size, args.img_size)
    # Fix argparse hyphen attr
    data_dir = args.data_dir

    with open("src/labels.json", "w") as f:
        json.dump(classes, f)

    model = build_model(args.model, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
        train_acc = correct / total
        train_loss = loss_sum / total

        # validation
        y_true, y_pred = evaluate(model, val_dl, device)
        val_acc = (y_true == y_pred).mean()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("models").mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), "models/derma_scanner_efficientnet_b0.pth")
            print("Saved new best model.")

    # final test metrics
    y_true, y_pred = evaluate(model, test_dl, device)
    print("Test accuracy:", (y_true == y_pred).mean())
    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    main()
