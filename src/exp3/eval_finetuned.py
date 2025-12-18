# ==========================================================
# eval_finetuned.py — Exp-3 (evaluación cross-dataset)
# Basado en Exp-4 con:
# Evalúa el modelo entrenado en AffectNet (Exp-3)
# Usa test sets: AffectNet, FER-2013, RAF-DB y CK+
# Guarda todos los CSVs y ZIP final en modelFER/exports/<backbone>/exp3/exports/
# ==========================================================

import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix
)

# ---- imports del proyecto ----
from dataloaders import (
    load_config, load_affectnet, load_rafdb,
    load_fer2013, load_ckplus
)
from models import build_mobilenetv3, build_efficientnet


# ==========================================================
# MÉTRICA DE CALIBRACIÓN (Expected Calibration Error)
# ==========================================================
def expected_calibration_error(probs, labels, n_bins=15):
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


# ==========================================================
# FUNCIÓN DE EVALUACIÓN
# ==========================================================
def evaluate_model(model, dataloader, device, num_classes, dataset_name="testset", backbone="mobilenetv3", results_dir=""):
    model.eval()
    y_true, y_pred, y_probs, latencies = [], [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"[EVAL] {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)
    latencies = np.array(latencies)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    ece = expected_calibration_error(torch.tensor(y_probs), torch.tensor(y_true))
    p50, p95 = np.percentile(latencies, 50), np.percentile(latencies, 95)

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

    preds_df = pd.DataFrame(y_probs, columns=[f"class_{i}" for i in range(num_classes)])
    preds_df.insert(0, "y_true", y_true)
    preds_path = os.path.join(results_dir, f"preds_{dataset_name.lower()}_{backbone}.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"[INFO] Predicciones guardadas en {preds_path}")

    return metrics, f1_score(y_true, y_pred, average=None, labels=range(num_classes)), cm


def make_zip(zip_path, dirs):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    valid = [d for d in dirs if os.path.exists(d)]
    if not valid:
        return
    os.system(f'zip -r "{zip_path}" {" ".join(valid)}')


# ==========================================================
# MAIN
# ==========================================================
def main():
    cfg = load_config("/kaggle/working/modelFER/configs/phase3E3.1.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg["backbone"]
    num_classes = cfg["model"]["num_classes"]

    # --- Raíz del proyecto ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Directorios de resultados (interno y externo) ---
    results_dir = os.path.join(base_dir, "outputs", "phase3", backbone, "exp3", "exports")
    os.makedirs(results_dir, exist_ok=True)

    external_zip_dir = os.path.join(base_dir, "exports", backbone, "exp3", "exports")
    os.makedirs(external_zip_dir, exist_ok=True)
    zip_path = os.path.join(external_zip_dir, f"eval_results_{backbone}.zip")

    # --- Buscar checkpoint ---
    ckpt_path = os.path.join(base_dir, "outputs", "phase3", backbone, "exp3", "checkpoints", f"best_{backbone}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No se encontró checkpoint en {ckpt_path}")
    print(f"[INFO] Checkpoint cargado: {ckpt_path}")

    # --- Construir modelo ---
    if "mobilenetv3" in backbone.lower():
        model = build_mobilenetv3(num_classes=num_classes, pretrained=False)
    else:
        model = build_efficientnet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    # --- Cargar datasets de EVALUACIÓN ---
    aff_test, _ = load_affectnet("test", cfg)
    raf_test, _ = load_rafdb("test", cfg)
    fer_test, _ = load_fer2013("test", cfg)
    ck_test, _ = load_ckplus(cfg)  # CORREGIDO

    datasets = [
        (aff_test, "AffectNet"),
        (raf_test, "RAF-DB"),
        (fer_test, "FER-2013"),
        (ck_test, "CK+")
    ]

    all_metrics, all_per_class, all_confusion = [], {}, {}

    for loader, name in datasets:
        metrics, per_class_f1, cm = evaluate_model(
            model, loader, device, num_classes,
            dataset_name=name, backbone=backbone, results_dir=results_dir
        )
        all_metrics.append(metrics)
        all_per_class[name] = per_class_f1.tolist()
        all_confusion[name] = cm.tolist()

    # --- Guardar resúmenes ---
    pd.DataFrame(all_metrics).to_csv(os.path.join(results_dir, f"summary_{backbone}.csv"), index=False)
    pd.DataFrame(all_per_class).to_csv(os.path.join(results_dir, f"per_class_f1_{backbone}.csv"))
    pd.DataFrame({k: [v] for k, v in all_confusion.items()}).to_csv(
        os.path.join(results_dir, f"confusion_matrix_{backbone}.csv"), index=False
    )

    # --- ZIP externo (visible) ---
    try:
        make_zip(zip_path, [results_dir])
        print(f"[INFO] Resultados empaquetados en: {zip_path}")
    except Exception as e:
        print(f"[WARN] No se pudo crear el ZIP: {e}")

    print(f"\n[INFO] Evaluación completada. Resultados en {results_dir}")


if __name__ == "__main__":
    main()
