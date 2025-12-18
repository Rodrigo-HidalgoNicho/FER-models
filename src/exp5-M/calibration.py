import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

OUTPUT_DIR = "outputs/csv/phase3M"

def compute_calibration(y_true, y_pred_probs, n_bins=10):
    """
    Calcula curva de confiabilidad y ECE.
    """
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    accuracies = (predictions == y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1

    acc_per_bin, conf_per_bin, counts = [], [], []
    ece = 0.0

    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            acc_bin = np.mean(accuracies[mask])
            conf_bin = np.mean(confidences[mask])
            acc_per_bin.append(float(acc_bin))
            conf_per_bin.append(float(conf_bin))
            counts.append(int(np.sum(mask)))
            ece += (np.sum(mask) / len(y_true)) * abs(acc_bin - conf_bin)
        else:
            acc_per_bin.append(None)
            conf_per_bin.append(None)
            counts.append(0)

    return {
        "bins": bins.tolist(),
        "accuracy_per_bin": acc_per_bin,
        "confidence_per_bin": conf_per_bin,
        "counts": counts,
        "ECE": float(ece)
    }

def save_calibration(y_true, y_pred_probs, dataset_name, run_name="finetuned"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    calib = compute_calibration(y_true, y_pred_probs)
    out_path = os.path.join(OUTPUT_DIR, f"calibration_{run_name}_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[INFO] Calibration guardado en {out_path}")

# === Ejemplo de uso ===
if __name__ == "__main__":
    # Cargar desde CSVs de eval_finetuned (y_true, y_probs)
    # Suponemos que guardaste un archivo con predicciones
    preds_path = os.path.join(OUTPUT_DIR, "preds_affectnet.csv")
    if os.path.exists(preds_path):
        df = pd.read_csv(preds_path)
        y_true = df["y_true"].values
        y_pred_probs = df.drop(columns=["y_true"]).values
        save_calibration(y_true, y_pred_probs, "affectnet")
    else:
        print("[WARN] No se encontró preds_affectnet.csv. Corre eval_finetuned.py con guardado de predicciones.")
