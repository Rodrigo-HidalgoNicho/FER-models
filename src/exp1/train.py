import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from dataloaders import load_config, load_fer2013, load_affectnet
from models import build_mobilenetv3, build_efficientnet


# Focal Loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)
        p = torch.exp(-logp)
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()


# Build model

def build_model(cfg):
    backbone = cfg.get("backbone", "mobilenetv3")
    num_classes = cfg["model"]["num_classes"]

    if backbone == "mobilenetv3":
        print("[INFO] Usando backbone: MobileNetV3-Large")
        return build_mobilenetv3(num_classes=num_classes)
    elif backbone == "efficientnetb0":
        print("[INFO] Usando backbone: EfficientNet-B0")
        return build_efficientnet(num_classes=num_classes)
    else:
        raise ValueError(f"Backbone desconocido: {backbone}")


# Train loop

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, scheduler=None):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for _, (images, labels) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# Evaluate

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, f1, prec, rec


# Main

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("backbone", "efficientnetb0")

    print("[INFO] Cargando datasets...")
    train_loader, _ = load_fer2013("train", cfg)
    aff_train, _ = load_affectnet("train", cfg)
    val_loader, _ = load_affectnet("val", cfg)

    # Combinar FER + AffectNet
    train_loader.dataset.samples.extend(aff_train.dataset.samples)
    train_loader.dataset.targets.extend(aff_train.dataset.targets)

    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    criterion = FocalLoss(gamma=cfg["loss"]["gamma"]) if cfg["loss"]["type"] == "focal" \
        else nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # --- Paths organizados según backbone
    if backbone == "mobilenetv3":
        base_dir = os.path.join(cfg["experiment"]["output_dir"], "mobilenetv3", "exp1")
    elif backbone == "efficientnetb0":
        base_dir = os.path.join(cfg["experiment"]["output_dir"], "efficientnetb0", "corrida")
    else:
        raise ValueError(f"Backbone desconocido: {backbone}")

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    csv_dir = os.path.join(base_dir, "csv")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    best_ckpt = os.path.join(ckpt_dir, f"best_{backbone}.pt")
    final_ckpt = os.path.join(ckpt_dir, f"final_{backbone}.pt")
    history_csv = os.path.join(csv_dir, f"train_history_{backbone}.csv")

    # --- Entrenamiento
    best_f1, patience, wait = 0, 7, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_prec": [], "val_rec": []}
    total_epochs = cfg["train"]["epochs"]

    for epoch in range(1, total_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)

        print(f"[PHASE3] Epoch {epoch}/{total_epochs} "
              f"- loss {train_loss:.4f} acc {train_acc:.4f} "
              f"val_loss {val_loss:.4f} val_f1 {val_f1:.4f} "
              f"({time.time()-start:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"[INFO] Guardado {best_ckpt}")
        else:
            wait += 1
            if wait >= patience:
                print("[INFO] Early stopping activado.")
                break

    torch.save(model.state_dict(), final_ckpt)
    print(f"[INFO] Modelo final guardado en {final_ckpt}")
    pd.DataFrame(history).to_csv(history_csv, index=False)
    print(f"[INFO] Historial guardado en {history_csv}")

    # --- Exportar ZIP (igual que antes)
    export_dir = os.path.join("exports")
    os.makedirs(export_dir, exist_ok=True)
    zip_path = os.path.join(export_dir, f"phase3_train_{backbone}.zip")
    os.system(f"zip -r {zip_path} {ckpt_dir} {csv_dir}")
    print(f"[INFO] Resultados empaquetados en: {zip_path}")

if __name__ == "__main__":
    main()
