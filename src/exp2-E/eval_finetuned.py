import os, torch, json, shutil, time
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from dataloaders import load_config, load_affectnet, load_rafdb, load_fer2013, load_ckplus
from models import build_mobilenetv3, build_efficientnet
from calibration import save_calibration


def evaluate_dataset(model, dataloader, device, class_names, dataset_name, backbone, export_dir, phase="eval"):
    model.eval()
    y_true, y_pred, all_probs, latencies, records = [], [], [], [], []

    # Manejar Subset
    if isinstance(dataloader.dataset, torch.utils.data.Subset):
        samples = [dataloader.dataset.dataset.samples[i] for i in dataloader.dataset.indices]
    else:
        samples = dataloader.dataset.samples

    use_cuda_timing = torch.cuda.is_available()

    start_time = time.time()
    with torch.no_grad():
        for idx, ((img_path, _), (images, labels)) in enumerate(
            tqdm(zip(samples, dataloader), total=len(dataloader), desc=f"[{dataset_name}] Evaluando")
        ):
            images, labels = images.to(device), labels.to(device)

            # medir latencia
            if use_cuda_timing:
                ev_start = torch.cuda.Event(enable_timing=True); ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                outputs = model(images)
                ev_end.record(); torch.cuda.synchronize()
                latency = ev_start.elapsed_time(ev_end) / 1000.0
            else:
                t0 = time.time()
                outputs = model(images)
                latency = time.time() - t0

            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            confidence_pred = probs[range(len(labels)), preds].cpu().numpy()
            prob_true_class = probs[range(len(labels)), labels].cpu().numpy()

            for p, y_t, y_p, conf, p_true in zip([img_path], labels_np, preds, confidence_pred, prob_true_class):
                records.append({
                    "dataset": dataset_name,
                    "image_path": p,
                    "label_id": int(y_t),
                    "label_name": class_names[int(y_t)],
                    "pred_id": int(y_p),
                    "pred_name": class_names[int(y_p)],
                    "is_correct": int(y_t == y_p),
                    "confidence_pred": float(conf),
                    "prob_true_class": float(p_true),
                    "latency": latency
                })

            y_true.extend(labels_np)
            y_pred.extend(preds)
            all_probs.append(probs.cpu())
            latencies.append(latency)

    elapsed = time.time() - start_time
    print(f"[INFO] {dataset_name} completado en {elapsed:.2f} segundos")

    # convertir a arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    all_probs = torch.cat(all_probs, dim=0)
    latencies = np.array(latencies)

    # guardar detailed_results CSV
    detailed_path = os.path.join(export_dir, f"detailed_results_{dataset_name}_{backbone}_{phase}.csv")
    pd.DataFrame(records).to_csv(detailed_path, index=False)
    print(f"[INFO] Guardado {detailed_path}")

    return y_true, y_pred, all_probs, latencies


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg["model"]["backbone"]

    # cargar checkpoint fase 2
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_path = os.path.join(base_dir, "outputs", "phase3", backbone, "exp2",
                             "phase2", "checkpoints", f"best_multidataset_{backbone}.pt")
    model = build_mobilenetv3(num_classes=cfg["model"]["num_classes"], freeze=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[INFO] Loaded model from {ckpt_path}")

    # directorio export
    export_dir = os.path.join(base_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)

    datasets_eval = {
        "AffectNet": load_affectnet("test", cfg, phase="phase1")[0],
        "RAF-DB": load_rafdb("test", cfg, phase="phase1")[0],
        "FER-2013": load_fer2013("test", cfg, phase="phase2")[0],
        "CK+": load_ckplus(cfg, phase="phase2")[0]
    }
    class_names = cfg["classes"]

    summary_records, per_class_records = [], []

    for name, loader in datasets_eval.items():
        print(f"\n========== Evaluando {name} ==========")
        y_true, y_pred, probs, latencies = evaluate_dataset(
            model, loader, device, class_names,
            dataset_name=name, backbone=backbone,
            export_dir=export_dir, phase="eval"
        )

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)

        # latencia p50/p95
        p50, p95 = np.percentile(latencies, 50), np.percentile(latencies, 95)

        # guardar calibration
        calib_path = os.path.join(export_dir, f"calibration_{name}_{backbone}.json")
        ece = save_calibration(y_true, probs.numpy(), dataset_name=f"{name}_{backbone}", run_name="finetuned")

        # guardar confusion matrix
        cm_path = os.path.join(export_dir, f"confusion_matrix_{name}_{backbone}.csv")
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)

        # guardar per-class f1
        cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        for cls, metrics in cls_report.items():
            if cls in class_names:
                per_class_records.append({
                    "dataset": name,
                    "class": cls,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"]
                })

        # summary
        summary_records.append({
            "dataset": name,
            "accuracy": acc,
            "f1_macro": f1,
            "ece": ece,
            "latency_p50": p50,
            "latency_p95": p95
        })

    # guardar summary y per-class F1
    pd.DataFrame(summary_records).to_csv(
        os.path.join(export_dir, f"summary_{backbone}.csv"), index=False
    )
    pd.DataFrame(per_class_records).to_csv(
        os.path.join(export_dir, f"per_class_f1_{backbone}.csv"), index=False
    )

    # empaquetar resultados en ZIP
    zip_path = os.path.join(export_dir, f"eval_results_{backbone}.zip")
    os.system(f"zip -r {zip_path} {export_dir}")
    print(f"\n[INFO] Evaluation finished. Results packaged in: {zip_path}")


if __name__ == "__main__":
    main()
