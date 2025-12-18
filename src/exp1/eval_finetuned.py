import os
import time
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from dataloaders import load_config, load_fer2013, load_affectnet, load_ckplus
from models import build_mobilenetv3, build_efficientnet

# -------------------------------
# Calibration metric: ECE
# -------------------------------
def expected_calibration_error(probs, labels, n_bins=15):
    """Compute Expected Calibration Error (ECE)."""
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = confidences.gt(bin_boundaries[i]) * confidences.le(bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

# -------------------------------
# Eval function
# -------------------------------
def evaluate_model(model, dataloader, device, num_classes, dataset_name="testset", backbone="mobilenetv3", results_dir=""):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    latencies = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"[EVAL] {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.time()
            latencies.append(end - start)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    ece = expected_calibration_error(torch.tensor(y_probs), torch.tensor(y_true))
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    metrics = {
        "dataset": dataset_name,
        "backbone": backbone,
        "accuracy": acc,
        "f1_macro": f1,
        "precision": prec,
        "recall": rec,
        "ece": ece,
        "latency_p50": p50,
        "latency_p95": p95
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes))
    preds_df = pd.DataFrame(y_probs, columns=[f"class_{i}" for i in range(num_classes)])
    preds_df.insert(0, "y_true", y_true)
    preds_path = os.path.join(results_dir, f"preds_{dataset_name.lower()}_{backbone}.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"[INFO] Predicciones guardadas en {preds_path}")

    return metrics, per_class_f1, cm

# -------------------------------
# Main
# -------------------------------
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("backbone", "mobilenetv3")
    num_classes = cfg["model"]["num_classes"]

    # --- Paths organizados según backbone
    if backbone == "mobilenetv3":
        base_dir = os.path.join(cfg["experiment"]["output_dir"], "mobilenetv3", "exp1")
    elif backbone == "efficientnetb0":
        base_dir = os.path.join(cfg["experiment"]["output_dir"], "efficientnetb0", "exp1")
    else:
        raise ValueError(f"Backbone desconocido: {backbone}")

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "exports")
    os.makedirs(results_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"best_{backbone}.pt")

    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"No se encontró el checkpoint: {best_ckpt}")

    # --- Modelo
    if backbone == "mobilenetv3":
        model = build_mobilenetv3(num_classes=num_classes, pretrained=False)
    elif backbone == "efficientnetb0":
        model = build_efficientnet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.to(device)

    # --- Datasets (sin fases)
    aff_test, _ = load_affectnet("test", cfg)
    fer_test, _ = load_fer2013("test", cfg)
    ck_test, _ = load_ckplus(cfg)

    all_metrics, all_per_class, all_confusion = [], {}, {}

    for loader, name in [(aff_test, "AffectNet"), (fer_test, "FER-2013"), (ck_test, "CK+")]:
        metrics, per_class_f1, cm = evaluate_model(model, loader, device, num_classes, dataset_name=name, backbone=backbone, results_dir=results_dir)
        all_metrics.append(metrics)
        all_per_class[name] = per_class_f1.tolist()
        all_confusion[name] = cm.tolist()

    summary_csv = os.path.join(results_dir, f"summary_{backbone}.csv")
    perclass_csv = os.path.join(results_dir, f"per_class_f1_{backbone}.csv")
    cm_csv = os.path.join(results_dir, f"confusion_matrix_{backbone}.csv")

    pd.DataFrame(all_metrics).to_csv(summary_csv, index=False)
    pd.DataFrame(all_per_class).to_csv(perclass_csv)
    pd.DataFrame({k: [v] for k, v in all_confusion.items()}).to_csv(cm_csv, index=False)
    print(f"[INFO] Evaluación completada. Artefactos guardados en {results_dir}")

if __name__ == "__main__":
    main()
