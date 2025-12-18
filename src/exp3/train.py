# ==========================================================
# train.py — Exp-3 (AffectNet training único, evaluación cross-dataset)
# Basado en Exp-4 con:
# Solo AffectNet para entrenamiento (train + val)
# Evaluación con test sets de AffectNet, FER-2013, RAF-DB y CK+
# Sin teacher / KD
# Rutas: outputs/phase3/<backbone>/exp3/...
# ZIPs incrementales por mejora + ZIP final en modelFER/exports/
# ==========================================================

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from dataloaders import load_config, load_affectnet, load_rafdb, load_fer2013, load_ckplus
from models import build_mobilenetv3, build_efficientnet

# ==========================================================
# Monitoreo de recursos
# ==========================================================
def get_resource_status(start_time=None, batch_time=None):
    """Devuelve un dict con CPU/RAM/GPU/VRAM/TEMP (si hay) y latencia por batch."""
    cpu_pct = ram_pct = gpu_util = vram_pct = temp = None
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.0)
        ram_pct = psutil.virtual_memory().percent
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            gpu_util = util.gpu
            vram_pct = 100.0 * mem.used / max(1, mem.total)
            pynvml.nvmlShutdown()
        except Exception:
            try:
                gpu_util = None
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_used = torch.cuda.memory_allocated(0)
                vram_pct = 100.0 * vram_used / max(1, vram_total)
            except Exception:
                pass

    latency_ms = None
    if batch_time is not None:
        latency_ms = batch_time * 1000.0
    elif start_time is not None:
        latency_ms = (time.time() - start_time) * 1000.0

    return {
        "CPU": cpu_pct,
        "RAM": ram_pct,
        "GPU": gpu_util,
        "VRAM": vram_pct,
        "TEMP": temp,
        "LAT": latency_ms,
    }


def print_resource_warn(status, stable_tag="🟢 Estable"):
    def f(v, suf="%"):
        return "n/a" if v is None else (f"{v:.0f}{suf}" if suf else f"{v:.0f}")
    parts = [
        f"CPU: {f(status['CPU'])}",
        f"RAM: {f(status['RAM'])}",
        f"GPU: {f(status['GPU'])}",
        f"VRAM: {f(status['VRAM'])}",
        f"TEMP: {f(status['TEMP'], '°C')}",
        f"LAT: {f(status['LAT'], ' ms/batch')}",
        f"| {stable_tag}"
    ]
    print("[WARN] " + " | ".join(parts))

# ==========================================================
# Utilidades
# ==========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_zip(zip_path, dirs):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    valid = [d for d in dirs if os.path.exists(d)]
    if not valid:
        return
    os.system(f'zip -r "{zip_path}" {" ".join(valid)}')

# ==========================================================
# Entrenamiento / Evaluación
# ==========================================================
from torch.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, scheduler, device, criterion,
                    epoch, grad_clip_norm=None, warn_every=500, use_amp=True):
    model.train()
    scaler = GradScaler('cuda', enabled=use_amp)
    total_loss, total_correct, total_count = 0.0, 0, 0
    rolling_t = None

    for i, (images, labels) in enumerate(tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch+1}"), start=1):
        t0 = time.time()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and hasattr(scheduler, "step") and scheduler.__class__.__name__ == "OneCycleLR":
            scheduler.step()

        bs = labels.size(0)
        total_loss += float(loss) * bs
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_count += bs

        batch_time = time.time() - t0
        rolling_t = batch_time if rolling_t is None else 0.9 * rolling_t + 0.1 * batch_time
        if (i % warn_every) == 0:
            status = get_resource_status(batch_time=rolling_t)
            print_resource_warn(status)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, device, class_names, criterion_ce=None, export_csv_path=None):
    model.eval()
    y_true, y_pred, probs_all = [], [], []
    total_loss, total_count = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        if criterion_ce is not None:
            loss = criterion_ce(logits, labels)
            total_loss += float(loss) * labels.size(0)
            total_count += labels.size(0)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(probs.argmax(1).cpu().numpy().tolist())
        probs_all.append(probs.cpu())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    probs_all = torch.cat(probs_all, dim=0).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    if export_csv_path:
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(export_csv_path)

    val_loss = total_loss / max(1, total_count) if criterion_ce else None
    return acc, f1, cm, report, probs_all, y_true, y_pred, val_loss

# ==========================================================
# Main
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/kaggle/working/modelFER/configs/phase3E3.1.yaml", help="Ruta al YAML de configuración")
    parser.add_argument("--warn_every", type=int, default=500, help="Cada cuántos batches imprimir [WARN] recursos")
    args = parser.parse_args()

    cfg = load_config(args.config)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA disponible ✔ — GPU: {dev_name}")
    else:
        print("[INFO] CUDA NO disponible ✖ — ejecutando en CPU")

    device = torch.device("cuda" if use_cuda else "cpu")
    backbone = cfg["backbone"]

    print("[INFO] Cargando datasets (solo AffectNet para entrenamiento)...")
    train_loader, _ = load_affectnet("train", cfg)
    val_loader, _ = load_affectnet("val", cfg)

    # === Model ===
    if backbone == "mobilenetv3":
        model = build_mobilenetv3(num_classes=cfg["model"]["num_classes"]).to(device)
    else:
        model = build_efficientnet(num_classes=cfg["model"]["num_classes"]).to(device)

    # === Criterion / Optimizer / Scheduler ===
    label_smoothing = cfg["loss"].get("label_smoothing", 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["train"]["learning_rate"],
                            weight_decay=cfg["train"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    # === Directorios internos ===
    base_dir = os.path.join(cfg["experiment"]["output_dir"], backbone, "exp3")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    csv_dir = os.path.join(base_dir, "csv")
    for d in (ckpt_dir, csv_dir): ensure_dir(d)

    # === Directorio EXTERNO de exportación (VISIBLE) ===
    external_export_dir = os.path.join("modelFER", "exports", backbone, "exp3", "exports")
    ensure_dir(external_export_dir)

    best_ckpt_path = os.path.join(ckpt_dir, f"best_{backbone}.pt")
    hist_path = os.path.join(csv_dir, f"train_history_{backbone}.csv")
    cm_path = os.path.join(csv_dir, f"confusion_matrix_val_{backbone}.csv")
    zip_final = os.path.join(external_export_dir, f"exp3_results_{backbone}.zip")

    # === Entrenamiento ===
    best_f1, no_improve, patience, min_epochs = -1.0, 0, cfg["train"].get("patience", 8), cfg["train"].get("min_epochs", 10)
    history, epochs = [], cfg["train"]["epochs"]
    print("[INFO] Inicio del entrenamiento…")
    print_resource_warn(get_resource_status())

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion, epoch)
        val_acc, val_f1, cm, report, val_probs, y_true, y_pred, val_loss = evaluate(
            model, val_loader, device, [str(i) for i in range(cfg["model"]["num_classes"])],
            criterion_ce=criterion, export_csv_path=cm_path
        )

        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "time_sec": epoch_time
        })

        print(f"[EXP3] Epoch {epoch+1}/{epochs} | loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_f1 {val_f1:.4f} ({epoch_time:.1f}s)")

        # --- Mejor modelo ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[INFO] Guardado best model (val_f1={best_f1:.4f})")

            # --- ZIP incremental por mejora ---
            zip_improve = os.path.join(external_export_dir, f"best_{backbone}_{epoch+1}.zip")
            try:
                make_zip(zip_improve, [best_ckpt_path, csv_dir])
                print(f"[INFO] ZIP de mejora generado: {zip_improve}")
            except Exception as e:
                print(f"[WARN] No se pudo crear ZIP de mejora: {e}")

            no_improve = 0
        else:
            no_improve += 1

        pd.DataFrame(history).to_csv(hist_path, index=False)

        if (epoch + 1) >= min_epochs and no_improve >= patience:
            print(f"[INFO] Early stopping (F1 no mejora en {patience} epochs consecutivos).")
            break

    # --- ZIP final completo ---
    make_zip(zip_final, [ckpt_dir, csv_dir])
    print(f"[INFO] Entrenamiento completo y resultados empaquetados en {zip_final}")


if __name__ == "__main__":
    main()
