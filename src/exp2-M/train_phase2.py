import os, torch, pandas as pd, time, json, sys, psutil, matplotlib.pyplot as plt
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from dataloaders import load_config, load_fer2013
from models import build_mobilenetv3, build_efficientnet
from torchvision.models import resnet50, resnet101


# GPU monitoring seguro (compatible CPU / GPU)

USE_GPU = False
gpu_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _ = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    USE_GPU = True
except Exception as e:
    print(f"[WARN] Monitoreo de GPU deshabilitado ({e}). Ejecutando solo con CPU.")
    USE_GPU = False


def get_resource_status(start_time=None, batch_time=None):
    """Retorna consumo de CPU, GPU, RAM, VRAM y estado."""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    gpu_util = vram_usage = temp = 0
    global USE_GPU
    if USE_GPU and gpu_handle is not None:
        try:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            vram_usage = (meminfo.used / meminfo.total) * 100
            temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            print("[WARN] Error en monitoreo de GPU. Deshabilitando métricas GPU.")
            USE_GPU = False

    max_usage = max(cpu_usage, ram_usage, gpu_util, vram_usage)
    if max_usage < 70:
        status = "Estable"
    elif max_usage < 85:
        status = "Consumo alto"
    else:
        status = "Riesgo de término anticipado"

    latency = f"{batch_time*1000:.0f} ms/batch" if batch_time else "-"
    elapsed = f"{(time.time()-start_time)/60:.1f} min" if start_time else "-"

    msg = f"CPU: {cpu_usage:.0f}% | RAM: {ram_usage:.0f}%"
    if USE_GPU:
        msg += f" | GPU: {gpu_util:.0f}% | VRAM: {vram_usage:.0f}% | TEMP: {temp}°C"
    msg += f" | LAT: {latency} | {status} | Time: {elapsed}"
    return msg



# KD híbrida (Response + Relation-Based)

class DistillationLoss(nn.Module):
    def __init__(self, teacher, cfg, weight=None):
        super().__init__()
        self.teacher = teacher
        kd_cfg = cfg["knowledge_distillation"]
        self.alpha = kd_cfg.get("alpha", 0.5)
        self.beta = kd_cfg.get("beta", 0.3)
        self.gamma = kd_cfg.get("gamma", 0.2)
        self.temperature = kd_cfg.get("temperature", 4.0)
        self.use_distance = kd_cfg.get("use_distance", True)
        self.use_angle = kd_cfg.get("use_angle", True)

        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=cfg["loss"]["label_smoothing"]
        )

    def forward(self, student_outputs, teacher_outputs, targets,
                student_features=None, teacher_features=None):
        T = self.temperature

        kd_loss = F.kl_div(
            F.log_softmax(student_outputs / T, dim=1),
            F.softmax(teacher_outputs / T, dim=1),
            reduction="batchmean"
        ) * (T * T)

        ce_loss = self.ce(student_outputs, targets)

        def rkd_distance_loss(f_s, f_t):
            with torch.no_grad():
                td = F.pdist(f_t, p=2)
                td = td / (td.mean() + 1e-6)
            sd = F.pdist(f_s, p=2)
            sd = sd / (sd.mean() + 1e-6)
            return F.smooth_l1_loss(sd, td)

        def rkd_angle_loss(f_s, f_t):
            with torch.no_grad():
                td = (f_t.unsqueeze(0) - f_t.unsqueeze(1))
                tn = F.normalize(td, p=2, dim=2)
                ta = torch.bmm(tn, tn.transpose(1, 2)).view(-1)
            sd = (f_s.unsqueeze(0) - f_s.unsqueeze(1))
            sn = F.normalize(sd, p=2, dim=2)
            sa = torch.bmm(sn, sn.transpose(1, 2)).view(-1)
            return F.smooth_l1_loss(sa, ta)

        rkd_loss = 0.0
        if student_features is not None and teacher_features is not None:
            if self.use_distance:
                rkd_loss += rkd_distance_loss(student_features, teacher_features)
            if self.use_angle:
                rkd_loss += rkd_angle_loss(student_features, teacher_features)

        total_loss = (self.alpha * ce_loss) + (self.beta * kd_loss) + (self.gamma * rkd_loss)
        return total_loss



# Teacher

def build_teacher(teacher_name, num_classes):
    if teacher_name == "resnet50":
        print("[INFO] Teacher: ResNet-50")
        model = resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif teacher_name == "resnet101":
        print("[INFO] Teacher: ResNet-101")
        model = resnet101(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Teacher desconocido: {teacher_name}")



# Logging detallado

def log_detailed_results(model, teacher, dataloader, device, backbone, phase, dataset_name, output_dir, class_map):
    model.eval(); teacher.eval()
    records = []
    if hasattr(dataloader.dataset, "samples"):
        samples = dataloader.dataset.samples
    else:
        samples = []

    with torch.no_grad():
        for (img_path, _), (images, labels) in zip(samples, dataloader):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)
            end = time.time()
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            confidence_pred = probs[range(len(labels)), preds].cpu().numpy()
            prob_true_class = probs[range(len(labels)), labels].cpu().numpy()
            latency = end - start
            for p, y_true, y_pred, conf, p_true in zip([img_path], labels_np, preds, confidence_pred, prob_true_class):
                records.append({
                    "dataset": dataset_name,
                    "image_path": p,
                    "label_id": int(y_true),
                    "label_name": class_map.get(int(y_true), str(y_true)),
                    "pred_id": int(y_pred),
                    "pred_name": class_map.get(int(y_pred), str(y_pred)),
                    "is_correct": int(y_true == y_pred),
                    "confidence_pred": float(conf),
                    "prob_true_class": float(p_true),
                    "latency": latency
                })

    if records:
        df = pd.DataFrame(records)
        save_path = os.path.join(output_dir, f"detailed_results_{dataset_name}_{backbone}_{phase}.csv")
        df.to_csv(save_path, index=False)
        print(f"[INFO] Guardado CSV detallado: {save_path}")



# Entrenamiento con monitoreo

def train_one_epoch(model, teacher, dataloader, criterion, optimizer, device, epoch, total_epochs, scheduler=None, plateau=None):
    model.train(); running_loss, correct, total = 0,0,0
    start_time = time.time()
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs}", leave=True, ncols=100)

    for i, (images, labels) in enumerate(pbar):
        batch_start = time.time()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        with torch.no_grad(): teacher_outputs = teacher(images)
        loss = criterion(outputs, teacher_outputs, labels, student_features=outputs, teacher_features=teacher_outputs)
        loss.backward(); optimizer.step()
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        if plateau: plateau.step(loss.item())
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)

        if i % 500 == 0:
            batch_time = time.time() - batch_start
            status = get_resource_status(start_time, batch_time)
            sys.stdout.write("\033[F")
            sys.stdout.write(f"[WARN] {status.ljust(100)}\n")
            sys.stdout.flush()

    pbar.close(); print()
    return running_loss/total, correct/total



# Evaluación

def evaluate(model, teacher, dataloader, criterion, device):
    model.eval(); running_loss, correct, total = 0,0,0; y_true,y_pred = [],[]
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); teacher_outputs = teacher(images)
            loss = criterion(outputs, teacher_outputs, labels, student_features=outputs, teacher_features=teacher_outputs)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
            y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
    return running_loss/total, correct/total, f1_score(y_true,y_pred,average="macro"), precision_score(y_true,y_pred,average="macro",zero_division=0), recall_score(y_true,y_pred,average="macro",zero_division=0)



# Gráfico comparativo automático

def plot_training_curves(csv_dir, backbone):
    try:
        phase1_csv = os.path.join(csv_dir.replace("phase2", "phase1"), f"train_history_phase1_{backbone}.csv")
        phase2_csv = os.path.join(csv_dir, f"train_history_phase2_{backbone}.csv")

        if not (os.path.exists(phase1_csv) and os.path.exists(phase2_csv)):
            print("[WARN] No se encontraron CSVs de ambas fases para graficar.")
            return

        df1, df2 = pd.read_csv(phase1_csv), pd.read_csv(phase2_csv)
        plt.figure(figsize=(10,6))
        plt.suptitle(f"Evolución entre Fases ({backbone.upper()})", fontsize=14, weight='bold')

        plt.subplot(2,1,1)
        plt.plot(df1["val_f1"], label="Val F1 (Fase 1)", color="orange", linestyle="--")
        plt.plot(df2["val_f1"], label="Val F1 (Fase 2)", color="red")
        plt.axvline(x=len(df1["val_f1"]), color="gray", linestyle="--", label="Fin Fase 1")
        plt.ylabel("F1-macro"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)

        plt.subplot(2,1,2)
        plt.plot(df1["train_loss"], label="Train Loss (Fase 1)", color="blue", linestyle="--")
        plt.plot(df2["train_loss"], label="Train Loss (Fase 2)", color="navy")
        plt.axvline(x=len(df1["train_loss"]), color="gray", linestyle="--", label="Fin Fase 1")
        plt.xlabel("Épocas"); plt.ylabel("Pérdida"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout(rect=[0,0,1,0.97])
        out_path = os.path.join(csv_dir, f"training_comparison_{backbone}.png")
        plt.savefig(out_path); plt.close()
        print(f"[INFO] Gráfico comparativo guardado en {out_path}")
    except Exception as e:
        print(f"[WARN] Error al generar gráfico comparativo: {e}")



# Main

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg["model"]["backbone"]

    student = build_mobilenetv3(num_classes=cfg["model"]["num_classes"], freeze=False, unfreeze_blocks=2).to(device)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    phase1_ckpt = os.path.join(base_dir, "outputs", "phase3", backbone, "exp2", "phase1", "checkpoints", f"best_affectnet_raf_{backbone}.pt")
    student.load_state_dict(torch.load(phase1_ckpt, map_location=device))
    print(f"[INFO] Cargado checkpoint Phase1: {phase1_ckpt}")

    teacher_name = cfg["phases"]["phase2"].get("teacher", "resnet50")
    teacher = build_teacher(teacher_name, cfg["model"]["num_classes"]).to(device)
    teacher.eval()

    fer_train, _ = load_fer2013("train", cfg, phase="phase2")
    val_loader, _ = load_fer2013("test", cfg, phase="phase2")
    train_loader = fer_train

    lr = cfg["phases"]["phase2"].get("learning_rate", cfg["train"]["learning_rate"])
    weight_decay = cfg["train"].get("weight_decay", 0.0)
    if cfg["train"]["optimizer"] == "adamw":
        optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    elif cfg["train"]["optimizer"] == "sgd":
        optimizer = optim.SGD(student.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError("Optimizer desconocido")

    scheduler, plateau = None, None
    if cfg["train"]["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["phases"]["phase2"]["max_epochs"])
    elif cfg["train"]["scheduler"] == "plateau":
        plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg["train"].get("plateau_factor",0.5), patience=cfg["train"].get("plateau_patience",3))

    criterion = DistillationLoss(teacher, cfg, weight=torch.ones(cfg["model"]["num_classes"]).to(device))

    ckpt_dir = os.path.join(base_dir, "outputs", "phase3", backbone, "exp2", "phase2", "checkpoints")
    csv_dir  = os.path.join(base_dir, "outputs", "phase3", backbone, "exp2", "phase2", "csv")
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(csv_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"best_multidataset_{backbone}.pt")
    best_f1, best_epoch, best_val_loss = 0, -1, None

    history_records = []

    for epoch in range(cfg["phases"]["phase2"]["max_epochs"]):
        train_loss, train_acc = train_one_epoch(student, teacher, train_loader, criterion, optimizer, device, epoch+1, cfg["phases"]["phase2"]["max_epochs"], scheduler, plateau)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(student, teacher, val_loader, criterion, device)
        history_records.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec
        })
        print(f"[PHASE2] Epoch {epoch+1} - loss {train_loss:.4f} acc {train_acc:.4f} val_loss {val_loss:.4f} val_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch, best_val_loss = val_f1, epoch+1, val_loss
            torch.save(student.state_dict(), best_ckpt)
            print(f"[INFO] Guardado {best_ckpt}")

    hist_path = os.path.join(csv_dir, f"train_history_phase2_{backbone}.csv")
    pd.DataFrame(history_records).to_csv(hist_path, index=False)
    calib_path = os.path.join(csv_dir, f"calibration_phase2_{backbone}.json")
    with open(calib_path, "w") as f:
        json.dump({
            "backbone": backbone,
            "phase": "phase2",
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "best_val_loss": best_val_loss,
            "ckpt_path": best_ckpt
        }, f, indent=2)

    print(f"[INFO] Phase2 terminado. Best ckpt: {best_ckpt}")

    class_map = {i: c for i,c in enumerate(cfg["classes"])}
    fer_val, _ = load_fer2013("test", cfg, phase="phase2")
    log_detailed_results(student, teacher, fer_val, device, backbone, "phase2", "FER-2013", csv_dir, class_map)

    export_dir = os.path.join(base_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    phase1_ckpt_dir = os.path.join(base_dir, "outputs", "phase3", backbone, "phase1", "checkpoints")
    phase1_csv_dir  = os.path.join(base_dir, "outputs", "phase3", backbone, "phase1", "csv")

    zip_path = os.path.join(export_dir, f"phase2_results_{backbone}.zip")
    os.system(f"zip -r {zip_path} {ckpt_dir} {csv_dir} {phase1_ckpt_dir} {phase1_csv_dir}")
    print(f"[INFO] Resultados Fase 2 empaquetados en: {zip_path}")

    # Generar gráfico comparativo automático
    plot_training_curves(csv_dir, backbone)


if __name__ == "__main__":
    main()
