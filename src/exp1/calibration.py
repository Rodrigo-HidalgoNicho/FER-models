import os
import json
import numpy as np
import pandas as pd

def compute_calibration(y_true, y_pred_probs, n_bins=10):
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
    return {"bins": bins.tolist(), "accuracy_per_bin": acc_per_bin, "confidence_per_bin": conf_per_bin, "counts": counts, "ECE": float(ece)}

def process_backbone(backbone):
    if backbone == "mobilenetv3":
        input_dir = "outputs/phase3/mobilenetv3/exp1/exports"
        output_dir = "outputs/phase3/mobilenetv3/exp1/calibration"
    elif backbone == "efficientnetb0":
        input_dir = "outputs/phase3/efficientnetb0/exp1/exports"
        output_dir = "outputs/phase3/efficientnetb0/exp1/calibration"
    else:
        return
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.startswith("preds_") and f.endswith(".csv")]
    if not files:
        print(f"[WARN] No se encontraron archivos preds_*.csv en {input_dir}")
        return
    for fname in files:
        path = os.path.join(input_dir, fname)
        print(f"[INFO] Procesando {path}")
        df = pd.read_csv(path)
        y_true = df["y_true"].values
        y_pred_probs = df.drop(columns=["y_true"]).values
        dataset_name = fname.split("_")[1]
        calib = compute_calibration(y_true, y_pred_probs)
        out_path = os.path.join(output_dir, f"calibration_{dataset_name.lower()}_{backbone}.json")
        with open(out_path, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"[INFO] Calibration guardado en {out_path}")

if __name__ == "__main__":
    for b in ["mobilenetv3", "efficientnetb0"]:
        process_backbone(b)
