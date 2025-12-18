# train.py  —  Exp-5 (KD híbrida ligera: KL + RKD-Distance, single-phase, teacher único)
# - TODO sale del YAML (phase5M.yaml con --config).
# - Backbones: mobilenetv3, efficientnetv2_m (definidos en models.py).
# - Teacher único: ResNet-50 (definido en YAML).
# - Artefactos SIEMPRE en: modelFER/outputs/phase3/<backbone>/exp5/{checkpoints,csv,exports,calibration}
# - ZIPs fuera de outputs/: exports/<backbone>/exp5/exports/...
# - Incluye: tqdm, AMP (torch.amp), resume, early-stopping por F1, sampler balanceado, monitor recursos, CSV por epoch.
# - ZIP por mejora (best_1.zip, best_2.zip, ...) con checkpoint ligero + CSV; y ZIP final ligero.
# - CORRECCIÓN: RAF-DB val NO usa load_rafdb("val"); el split 90/10 se hace EN MEMORIA → no crea _splits en disco.

import os
import json
import time
import math
import random
import shutil
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm  # barra de progreso

# --- Proyecto: módulos locales ---
from dataloaders import load_config, load_affectnet, load_rafdb
try:
    from dataloaders import compute_class_weights
except Exception:
    compute_class_weights = None

from models import build_mobilenetv3, build_efficientnetv2_m
from calibration import save_calibration

from torchvision import models as tv_models
from torchvision.models import ResNet50_Weights



# Monitoreo de recursos

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

    status = {
        "CPU": cpu_pct,
        "RAM": ram_pct,
        "GPU": gpu_util,
        "VRAM": vram_pct,
        "TEMP": temp,
        "LAT": latency_ms,
    }
    return status


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



# Utilidades varias

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_student(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    b = backbone.lower()
    if b in ("mobilenetv3", "mobilenet_v3", "mobilenetv3-large", "mobilenet_v3_large"):
        return build_mobilenetv3(num_classes=num_classes, pretrained=pretrained)
    elif b in ("efficientnetv2_m", "efficientnet-v2-m", "efficientnet_v2_m"):
        return build_efficientnetv2_m(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Backbone no soportado: {backbone}")


def build_teacher(name: str = "resnet50", pretrained: bool = True, num_classes: int = 7) -> nn.Module:
    n = name.lower()
    if n == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        teacher = tv_models.resnet50(weights=weights)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher
    else:
        raise ValueError(f"Teacher no soportado: {name}")


# ---------- NUEVO: extracción de features (GAP 2048) del ResNet-50 ----------
@torch.no_grad()
def extract_resnet50_features(resnet50: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Reproduce el forward hasta avgpool del ResNet-50 de torchvision.
    Salida: tensor (B, 2048).
    """
    m = resnet50
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    x = m.avgpool(x)              # (B, 2048, 1, 1)
    x = torch.flatten(x, 1)       # (B, 2048)
    return x



# Distillation Loss (HÍBRIDA: KL + RKD-Distance + CE)

class DistillationLoss(nn.Module):
    def __init__(self, kd_cfg, ce_weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = kd_cfg.get("alpha", 0.55)      # CE
        self.beta = kd_cfg.get("beta", 0.30)        # KL (Response KD)
        self.gamma = kd_cfg.get("gamma", 0.40)      # RKD distance
        self.temperature = kd_cfg.get("temperature", 4.0)
        self.use_distance = kd_cfg.get("use_distance", True)
        self.use_angle = kd_cfg.get("use_angle", False)

        self.ce = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=label_smoothing)

    def rkd_distance_loss(self, f_s, f_t):
        # pairwise distances, normalized by mean distance
        with torch.no_grad():
            td = F.pdist(f_t, p=2)
            td = td / (td.mean() + 1e-6)
        sd = F.pdist(f_s, p=2)
        sd = sd / (sd.mean() + 1e-6)
        return F.smooth_l1_loss(sd, td)

    def forward(self, student_logits, teacher_logits, targets,
                student_features=None, teacher_features=None):
        T = self.temperature

        # CE (hard labels)
        ce_loss = self.ce(student_logits, targets)

        # KL (Response KD) — ahora ambos son (B, 7)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean"
        ) * (T * T)

        # Relation KD (distance)
        rkd_loss = 0.0
        if (student_features is not None) and (teacher_features is not None) and self.use_distance:
            rkd_loss = rkd_loss + self.rkd_distance_loss(student_features, teacher_features)

        total = (self.alpha * ce_loss) + (self.beta * kd_loss) + (self.gamma * rkd_loss)
        return total, {"ce": float(ce_loss), "kd": float(kd_loss), "rkd": float(rkd_loss)}



# Helpers para targets (mezcla) y ZIP

def _collect_targets(ds):
    """Extrae la lista de targets de un dataset arbitrario (ImageFolder / Subset / ConcatDataset)."""
    if isinstance(ds, ConcatDataset):
        t = []
        for c in ds.datasets:
            t.extend(_collect_targets(c))
        return t
    if isinstance(ds, Subset):
        base = ds.dataset
        idxs = ds.indices
        base_targets = _collect_targets(base)
        return [base_targets[i] for i in idxs]
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "samples"):
        return [label for _, label in ds.samples]
    raise ValueError("No pude extraer targets de este dataset (ni .targets ni .samples).")


def make_zip(zip_path, dirs):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    valid = [d for d in dirs if os.path.exists(d)]
    if len(valid) == 0:
        return
    if os.path.exists(zip_path):
        os.remove(zip_path)
    cmd = 'zip -r "{}" {}'.format(zip_path, " ".join('"{}"'.format(p) for p in valid))
    ret = os.system(cmd)
    if ret != 0:
        import tempfile, shutil as _sh
        base = os.path.splitext(zip_path)[0]
        with tempfile.TemporaryDirectory() as tmpd:
            for d in valid:
                _sh.copytree(d, os.path.join(tmpd, os.path.basename(d)))
            _sh.make_archive(base, 'zip', tmpd)


def next_zip_path(zip_dir, backbone, idx):
    """Construye un nombre único por mejora: exports/<backbone>/exp5/exports/best_<backbone>_<idx>.zip"""
    os.makedirs(zip_dir, exist_ok=True)
    return os.path.join(zip_dir, f"best_{backbone}_{idx}.zip")



# Entrenamiento / Validación

from torch.amp import autocast, GradScaler  # AMP nueva API

def train_one_epoch(model, teacher, teacher_adapter, loader, optimizer, scheduler, device, criterion,
                    epoch, kd_warmup_epochs=2, grad_clip_norm=None,
                    warn_every=500, use_amp=False):
    model.train()
    teacher.eval()  # backbone congelado

    scaler = GradScaler('cuda', enabled=use_amp)

    total_loss, total_correct, total_count = 0.0, 0, 0
    rolling_t = None

    # BARRA DE PROGRESO (estilo train_phase1.py)

    for i, (images, labels) in enumerate(
        tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {epoch+1}")
        , start=1
    ):
        t0 = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            # Student forward
            logits_s = model(images)

            # Teacher features (backbone sin grad) + adapter (con grad)
            with torch.no_grad():
                feat_t = extract_resnet50_features(teacher, images)  # (B, 2048)
            logits_t = teacher_adapter(feat_t)  # (B, num_classes) — grad SOLO para adapter

            # KD warm-up: sin KD híbrida los primeros epochs
            use_kd = (epoch + 1) > kd_warmup_epochs
            if use_kd:
                # RKD usa distancias par-a-par; dims pueden diferir (ok).
                loss, _parts = criterion(
                    student_logits=logits_s,
                    teacher_logits=logits_t,
                    targets=labels,
                    student_features=logits_s,
                    teacher_features=feat_t
                )
            else:
                # solo CE
                loss = criterion.ce(logits_s, labels)

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
        total_correct += (logits_s.argmax(1) == labels).sum().item()
        total_count += bs

        # --- monitoreo periódico estilo [WARN] ---
        batch_time = time.time() - t0
        rolling_t = batch_time if rolling_t is None else 0.9 * rolling_t + 0.1 * batch_time
        if (i % warn_every) == 0:
            status = get_resource_status(batch_time=rolling_t)
            print_resource_warn(status)

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, device, class_names, criterion_ce=None, export_csv_path=None):
    """
    Evalúa y (opcionalmente) calcula val_loss con la CE del 'criterion'.
    """
    model.eval()
    y_true, y_pred, probs_all = [], [], []
    total_loss, total_count = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        if criterion_ce is not None:
            loss = criterion_ce(logits, labels)
            total_loss += float(loss) * labels.size(0)
            total_count += labels.size(0)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(probs.argmax(1).cpu().numpy().tolist())
        probs_all.append(probs.cpu())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs_all = torch.cat(probs_all, dim=0).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    if export_csv_path:
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(export_csv_path)

    val_loss = (total_loss / max(1, total_count)) if criterion_ce is not None else None
    return acc, f1, cm, report, probs_all, y_true, y_pred, val_loss


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/kaggle/working/modelFER/configs/phase5M.yaml", help="Ruta al YAML de configuración")
    parser.add_argument("--warn_every", type=int, default=500, help="Cada cuántos batches imprimir [WARN] recursos")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(42)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA disponible ✔ — GPU: {dev_name}")
    else:
        print("[INFO] CUDA NO disponible ✖ — ejecutando en CPU")

    device = torch.device("cuda" if use_cuda else "cpu")

    # --- YAML fields ---
    exp_name     = cfg["experiment"]["name"]          # "phase5M"
    out_root_raw = cfg["experiment"]["output_dir"]    # "outputs/phase3"
    backbone     = cfg["model"]["backbone"]
    num_classes  = int(cfg["model"]["num_classes"])
    class_names  = cfg["classes"]

    # ► Anclar rutas a la RAÍZ del repo (…/modelFER)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def to_project_path(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(PROJECT_ROOT, p))

    # RUTA BASE de outputs anclada a modelFER/outputs/phase3
    out_root = to_project_path(out_root_raw)                           # …/modelFER/outputs/phase3
    base_out_dir = os.path.join(out_root, backbone, "exp5")            # …/modelFER/outputs/phase3/<backbone>/exp5

    # Subcarpetas estándar
    ckpt_dir        = os.path.join(base_out_dir, "checkpoints")
    csv_dir         = os.path.join(base_out_dir, "csv")
    export_dir      = os.path.join(base_out_dir, "exports")
    calibration_dir = os.path.join(base_out_dir, "calibration")
    for d in (ckpt_dir, csv_dir, export_dir, calibration_dir):
        ensure_dir(d)

    # Archivos
    best_ckpt_path   = os.path.join(ckpt_dir, f"best_affectnet_raf_{backbone}.pt")  # completo (reanudación)
    hist_path        = os.path.join(csv_dir, f"train_history_{backbone}.csv")
    cm_path          = os.path.join(csv_dir, f"confusion_matrix_val_{backbone}.csv")
    calib_path       = os.path.join(calibration_dir, f"calibration_{backbone}.json")

    # ZIPs EXTERNOS (fuera de outputs/):
    external_zip_dir = to_project_path(os.path.join("exports", backbone, "exp5", "exports"))
    ensure_dir(external_zip_dir)
    zip_path = os.path.join(external_zip_dir, f"exp5_results_{backbone}.zip")

    # --- Modelos ---
    student = build_student(backbone, num_classes=num_classes, pretrained=True).to(device)
    teacher_name = cfg.get("teacher", {}).get("name", "resnet50")
    teacher = build_teacher(teacher_name, pretrained=True, num_classes=num_classes).to(device)
    teacher.eval()

    # NUEVO: adapter 2048 -> num_classes para el teacher (se entrena)
    teacher_adapter = nn.Linear(2048, num_classes).to(device)

    # Cargar init_checkpoint (solo pesos del student).
    # Si NO se especifica en YAML, AUTO-apuntamos al best de Exp-4:
    init_ckpt = cfg.get("train", {}).get("init_checkpoint", None)
    if not init_ckpt:
        init_ckpt = os.path.join(out_root_raw, backbone, "exp4", "checkpoints", f"best_weights_{backbone}.pt")
    init_ckpt_abs = to_project_path(init_ckpt)

    if init_ckpt_abs and os.path.exists(init_ckpt_abs):
        try:
            sd = torch.load(init_ckpt_abs, map_location=device)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            student.load_state_dict(sd, strict=False)
            print(f"[INFO] Cargado init_checkpoint para student: {init_ckpt_abs}")
        except Exception as e:
            print(f"[WARN] No se pudo cargar init_checkpoint: {e}")
    else:
        print(f"[INFO] init_checkpoint no encontrado, arrancando desde pretrained ImageNet.")

    # --- Data (AffectNet + RAF-DB) ---
    aff_train_loader, aff_train_ds = load_affectnet("train", cfg)
    aff_val_loader,   aff_val_ds   = load_affectnet("val", cfg)

    # RAF-DB: SOLO train; split 90/10 EN MEMORIA
    _, raf_train_full = load_rafdb("train", cfg)
    raf_total = len(raf_train_full)
    raf_val_len = max(1, int(0.10 * raf_total))
    raf_train_len = raf_total - raf_val_len
    g = torch.Generator().manual_seed(42)
    raf_train_subset, raf_val_subset = random_split(raf_train_full, [raf_train_len, raf_val_len], generator=g)

    batch_size  = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 2))

    # DataLoaders para RAF
    raf_train_loader = DataLoader(raf_train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    raf_val_loader   = DataLoader(raf_val_subset,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

    # Fusionar datasets para un único loader (train y val)
    train_dataset = ConcatDataset([aff_train_ds, raf_train_subset])
    val_dataset   = ConcatDataset([aff_val_ds,   raf_val_subset])

    # Loader inicial (puede ser reemplazado por sampler balanceado)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # --- Class weights SOBRE LA MEZCLA (AffectNet + RAF train_subset)
    ce_weight = None
    try:
        fused_targets = torch.tensor(_collect_targets(train_dataset))
        num_classes_ = len(class_names)
        counts = torch.bincount(fused_targets, minlength=num_classes_).float()
        ce_weight = (1.0 / (counts + 1e-6))
        ce_weight = ce_weight / ce_weight.sum() * num_classes_
        ce_weight = ce_weight.to(device)
    except Exception as e:
        print(f"[WARN] No fue posible calcular class weights de la mezcla: {e}")
        ce_weight = None

    # --- (OPCIONAL) WeightedRandomSampler para balance por clase en train ---
    use_weighted_sampler = True  # pon en False si prefieres solo shuffle=True
    if use_weighted_sampler and ce_weight is not None:
        class_weights_np = ce_weight.detach().cpu().numpy()
        fused_targets_all = np.array(_collect_targets(train_dataset))
        sample_weights = class_weights_np[fused_targets_all]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(fused_targets_all),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    elif use_weighted_sampler:
        print("[WARN] No hay ce_weight → mantengo shuffle=True sin sampler.")

    # --- Criterion (KD híbrida)
    kd_cfg = cfg["knowledge_distillation"]
    label_smoothing = float(cfg["loss"].get("label_smoothing", 0.0))
    criterion = DistillationLoss(kd_cfg, ce_weight=ce_weight, label_smoothing=label_smoothing)

    # --- Optimizador / Scheduler
    lr = float(cfg["train"]["learning_rate"])
    wd = float(cfg["train"].get("weight_decay", 0.05))
    optimizer_name = cfg["train"].get("optimizer", "adamw").lower()

    # INCLUYE params del student + adapter del teacher (para que el KL sea 7 vs 7)
    params = list(student.parameters()) + list(teacher_adapter.parameters())

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        raise ValueError(f"Optimizer no soportado: {optimizer_name}")

    epochs = int(cfg["train"]["epochs"])
    scheduler_name = cfg["train"].get("scheduler", "onecycle").lower()
    if scheduler_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                  steps_per_epoch=len(train_loader),
                                                  epochs=epochs)
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor=float(cfg["train"].get("plateau_factor", 0.5)),
                                                         patience=int(cfg["train"].get("plateau_patience", 5)))
    else:
        scheduler = None

    grad_clip_norm   = cfg["train"].get("grad_clip_norm", None)
    kd_warmup_epochs = int(cfg["train"].get("kd_warmup_epochs", 2))

    # --- Parámetros de control
    patience   = int(cfg["train"].get("patience", 10))
    min_epochs = int(cfg["train"].get("min_epochs", 10))
    val_every  = int(cfg["train"].get("val_every", 1))
    use_amp    = bool(cfg["train"].get("amp", True))

    no_improve = 0
    best_f1 = -1.0
    history = []
    best_zip_idx = 0  # contador para zips por mejora

    print("[INFO] Inicio de entrenamiento…")
    status0 = get_resource_status()
    print_resource_warn(status0)

    # --- Resume automático desde "last"
    last_ckpt_path = os.path.join(ckpt_dir, f"last_{backbone}.pt")
    start_epoch = 0
    if os.path.exists(last_ckpt_path):
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device)
            if "model" in ckpt:
                student.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception:
                    pass
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            print(f"[INFO] Reanudando desde epoch {start_epoch} con last checkpoint.")
        except Exception as e:
            print(f"[WARN] No se pudo reanudar desde 'last': {e}")

    # --- Loop de entrenamiento
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            student, teacher, teacher_adapter, train_loader, optimizer, scheduler, device, criterion,
            epoch=epoch, kd_warmup_epochs=kd_warmup_epochs, grad_clip_norm=grad_clip_norm,
            warn_every=args.warn_every, use_amp=use_amp
        )

        do_validation = ((epoch + 1) % val_every == 0) or (epoch + 1 == epochs)
        if do_validation:
            val_acc, val_f1, cm, report, val_probs, y_true, y_pred, val_loss = evaluate(
                student, val_loader, device, class_names, criterion_ce=criterion.ce, export_csv_path=cm_path
            )
        else:
            val_acc = val_f1 = val_loss = None
            val_probs = y_true = y_pred = None

        if scheduler_name == "plateau" and scheduler is not None and do_validation and (val_f1 is not None):
            scheduler.step(val_f1)

        # --- Calibración: guardar SIEMPRE en .../outputs/phase3/<backbone>/exp5/calibration
        try:
            ece_val = None
            if do_validation and (val_probs is not None):
                ece_val = save_calibration(y_true, val_probs,
                                           dataset_name=f"val_{backbone}",
                                           run_name=exp_name,
                                           out_path=calib_path)
        except Exception:
            ece_val = None

        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": (None if val_loss is None else round(val_loss, 6)),
            "val_acc": (None if val_acc is None else round(val_acc, 6)),
            "val_f1": (None if val_f1 is None else round(val_f1, 6)),
            "ece": (None if ece_val is None else float(ece_val)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": round(epoch_time, 2),
        })

        # Log de resumen de epoch
        print(f"[TRAIN] Ep {epoch+1}/{epochs} | "
              f"loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {('n/a' if val_loss is None else f'{val_loss:.4f}')} "
              f"val_acc {('n/a' if val_acc is None else f'{val_acc:.4f}')} "
              f"val_f1 {('n/a' if val_f1 is None else f'{val_f1:.4f}')} | "
              f"ece {ece_val if ece_val is not None else 'n/a'} | "
              f"{epoch_time:.1f}s")

        # Guardar mejor por F1 (si hubo validación)
        if do_validation and (val_f1 is not None) and (val_f1 > best_f1):
            best_f1 = val_f1

            # 1) checkpoint completo (para reanudar) — NO va al ZIP
            torch.save({"model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                        "epoch": epoch}, best_ckpt_path)
            print(f"[INFO] Guardado best ckpt: {best_ckpt_path} (val_f1={best_f1:.4f})")

            # 2) checkpoint ligero (solo pesos) — este sí va al ZIP
            best_weights_path = os.path.join(ckpt_dir, f"best_weights_{backbone}.pt")
            torch.save(student.state_dict(), best_weights_path)

            # 3) ZIP por mejora: best_<idx>.zip (solo best_weights + csv/)
            best_zip_idx += 1
            per_improve_zip = next_zip_path(external_zip_dir, backbone, best_zip_idx)
            try:
                make_zip(per_improve_zip, [best_weights_path, csv_dir])
                print(f"[INFO] ZIP (mejora #{best_zip_idx}) generado: {per_improve_zip}")
            except Exception as e:
                print(f"[WARN] No se pudo crear ZIP de mejora: {e}")

            no_improve = 0
        else:
            no_improve += 1 if (do_validation and (epoch + 1) >= min_epochs) else 0

        # Guardar "last" SIEMPRE para reanudar (no va al ZIP)
        last_ckpt_path = os.path.join(ckpt_dir, f"last_{backbone}.pt")
        torch.save({"model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                    "epoch": epoch}, last_ckpt_path)

        # Guardar historia CSV (append-safe)
        pd.DataFrame(history).to_csv(hist_path, index=False)

        # Early stopping (por F1, con min_epochs)
        if (epoch + 1) >= min_epochs and no_improve >= patience:
            print(f"[INFO] Early stopping por F1 (patience={patience}). Mejor F1: {best_f1:.4f}")
            break

    # --- ZIP final ligero (fuera de outputs/) — usa el último best_weights + CSV
    try:
        best_weights_path = os.path.join(ckpt_dir, f"best_weights_{backbone}.pt")
        make_zip(zip_path, [best_weights_path, csv_dir])
        print(f"[INFO] Resultados empaquetados en: {zip_path}")
    except Exception as e:
        print(f"[WARN] No se pudo crear el ZIP final: {e}")


if __name__ == "__main__":
    main()
