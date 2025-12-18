import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------
# Config loader
# -------------------------------
def load_config(path="/kaggle/working/modelFER/configs/phase5M.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------
# Compute class weights
# -------------------------------
def compute_class_weights(dataset, num_classes):
    """Calcula pesos por clase a partir de dataset.targets (longitud = num_samples)."""
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=num_classes)
    weights = 1.0 / (counts.float() + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights

# -------------------------------
# MixUp / CutMix utils (si los usas en tu loop)
# -------------------------------
def mixup_data(x, y, alpha=0.2):
    """Aplica MixUp a un batch de imágenes."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Aplica CutMix a un batch de imágenes."""
    lam = np.random.beta(alpha, alpha)
    b, _, h, w = x.size()
    index = torch.randperm(b, device=x.device)

    cx, cy = np.random.randint(w), np.random.randint(h)
    cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
    x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
    x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)

    new_x = x.clone()
    new_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam

# -------------------------------
# Transforms (sin fases)
# -------------------------------
def get_transforms(cfg, split="train"):
    """Construye transforms según cfg['input'].size y cfg['input'].augmentations."""
    img_size = tuple(cfg["input"]["size"][:2])
    augs = set(cfg["input"].get("augmentations", []))

    pil_transforms, tensor_transforms = [], []

    if "randaugment" in augs and split == "train":
        pil_transforms.append(transforms.RandAugment(num_ops=2, magnitude=9))
    if "flip" in augs and split == "train":
        pil_transforms.append(transforms.RandomHorizontalFlip())
    if "randomcrop" in augs and split == "train":
        pil_transforms.append(transforms.RandomResizedCrop(img_size))

    # redimensionado/crop centrado (seguro para val/test)
    pil_transforms.extend([transforms.Resize(img_size), transforms.CenterCrop(img_size)])

    tensor_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if "randomerasing" in augs and split == "train":
        tensor_transforms.append(transforms.RandomErasing(p=0.25))

    return transforms.Compose(pil_transforms + tensor_transforms)

# -------------------------------
# Helper para DataLoader
# -------------------------------
def build_loader(dataset, cfg, split):
    num_workers = int(cfg["train"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

# -------------------------------
# Loaders
# -------------------------------
def load_affectnet(split, cfg, batch_size=None):
    """
    split ∈ {"train","val","test"} según hayas definido en el YAML de AffectNet.
    """
    ds_cfg = cfg["datasets"]["affectnet"]
    key = f"{split}_dir"
    if key not in ds_cfg:
        raise KeyError(f"[AffectNet] No se encontró '{key}' en datasets.affectnet del YAML.")
    ds_path = ds_cfg[key]

    transform = get_transforms(cfg, split)
    dataset = datasets.ImageFolder(root=ds_path, transform=transform)

    # Remapea clases al espacio común de 7 clases (con posible 'null' en AffectNet)
    mapping, class_names = cfg["affectnet_map"], cfg["classes"]
    # dataset.samples = [(path, target_idx_as_int_in_dataset.classes)]
    filtered_samples = []
    for p, idx in dataset.samples:
        # idx es índice en dataset.classes (que suelen ser strings '0'..'7')
        cls_name = dataset.classes[idx]
        # mapeamos por nombre si existe; si no, por índice como string
        mapped = mapping.get(str(cls_name), mapping.get(str(idx), None))
        if mapped is None:
            continue
        new_label = class_names.index(mapped)
        filtered_samples.append((p, new_label))

    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = class_names

    loader = DataLoader(
        dataset,
        batch_size=(batch_size or int(cfg["train"]["batch_size"])),
        shuffle=(split == "train"),
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True
    )
    return loader, dataset

def load_rafdb(split, cfg, batch_size=None):
    """
    RAF-DB:
      - Si el YAML define rafdb.val_dir -> usamos carpetas.
      - Si NO define val_dir -> hacemos split 90/10 (o cfg.datasets.rafdb.val_ratio) desde train.
    split ∈ {"train","val","test"}
    """
    ds_cfg = cfg["datasets"]["rafdb"]
    transform = get_transforms(cfg, split if split in ("train", "val") else "test")

    # --- Si hay test_dir y piden test, cargamos directo ---
    if split == "test":
        key = "test_dir"
        if key not in ds_cfg:
            raise KeyError(f"[RAF-DB] No se encontró '{key}' en datasets.rafdb del YAML.")
        ds_path = ds_cfg[key]
        dataset = datasets.ImageFolder(root=ds_path, transform=transform)

        # mapear a clases comunes
        mapping, class_names = cfg["rafdb_map"], cfg["classes"]
        filtered = []
        for p, idx in dataset.samples:
            cls_name = dataset.classes[idx]
            mapped = mapping.get(str(cls_name), None)
            if mapped is None: continue
            new_label = class_names.index(mapped)
            filtered.append((p, new_label))
        dataset.samples = filtered
        dataset.targets = [s[1] for s in filtered]
        dataset.classes = class_names

        loader = DataLoader(
            dataset,
            batch_size=(batch_size or int(cfg["train"]["batch_size"])),
            shuffle=False,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            pin_memory=True
        )
        return loader, dataset

    # --- train/val: con o sin val_dir ---
    has_val_dir = "val_dir" in ds_cfg and ds_cfg["val_dir"] and os.path.isdir(ds_cfg["val_dir"])
    if has_val_dir:
        key = f"{split}_dir"
        if key not in ds_cfg:
            raise KeyError(f"[RAF-DB] No se encontró '{key}' en datasets.rafdb del YAML.")
        ds_path = ds_cfg[key]
        base_ds = datasets.ImageFolder(root=ds_path, transform=transform)
    else:
        # Cargamos TODO train y partimos 90/10 (o val_ratio)
        train_dir = ds_cfg.get("train_dir", None)
        if train_dir is None:
            raise KeyError("[RAF-DB] Se esperaba 'train_dir' en datasets.rafdb del YAML.")
        base_full = datasets.ImageFolder(root=train_dir, transform=transform)

        # Mapeo a etiquetas comunes ANTES de partir (para estratificar por clase)
        mapping, class_names = cfg["rafdb_map"], cfg["classes"]
        filtered = []
        for p, idx in base_full.samples:
            cls_name = base_full.classes[idx]
            mapped = mapping.get(str(cls_name), None)
            if mapped is None: continue
            new_label = class_names.index(mapped)
            filtered.append((p, new_label))
        base_full.samples = filtered
        base_full.targets = [s[1] for s in filtered]
        base_full.classes = class_names

        # Estratificación simple por índice
        val_ratio = float(ds_cfg.get("val_ratio", 0.1))
        seed = int(ds_cfg.get("split_seed", 42))

        # Cache de índices para que train/val sean consistentes entre llamadas
        split_root = os.path.join(cfg["experiment"]["output_dir"], "_splits", "rafdb")
        os.makedirs(split_root, exist_ok=True)
        train_idx_file = os.path.join(split_root, "train_idx.npy")
        val_idx_file = os.path.join(split_root, "val_idx.npy")

        if os.path.exists(train_idx_file) and os.path.exists(val_idx_file):
            train_indices = np.load(train_idx_file)
            val_indices = np.load(val_idx_file)
        else:
            rng = np.random.RandomState(seed)
            targets = np.array(base_full.targets)
            train_indices, val_indices = [], []
            for c in np.unique(targets):
                idxs = np.where(targets == c)[0]
                rng.shuffle(idxs)
                n_val = max(1, int(round(len(idxs) * val_ratio)))
                val_indices.extend(idxs[:n_val].tolist())
                train_indices.extend(idxs[n_val:].tolist())
            train_indices = np.array(sorted(train_indices))
            val_indices = np.array(sorted(val_indices))
            np.save(train_idx_file, train_indices)
            np.save(val_idx_file, val_indices)

        from torch.utils.data import Subset
        train_ds = Subset(base_full, train_indices)
        val_ds = Subset(base_full, val_indices)

        # Según split, elegimos
        base_ds = train_ds if split == "train" else val_ds

    # Si llegamos aquí: base_ds ya es el dataset correcto (ImageFolder o Subset)
    # Si es ImageFolder de val_dir, aún no hemos remapeado → mapear ahora
    if isinstance(base_ds, datasets.ImageFolder):
        mapping, class_names = cfg["rafdb_map"], cfg["classes"]
        filtered = []
        for p, idx in base_ds.samples:
            cls_name = base_ds.classes[idx]
            mapped = mapping.get(str(cls_name), None)
            if mapped is None: continue
            new_label = class_names.index(mapped)
            filtered.append((p, new_label))
        base_ds.samples = filtered
        base_ds.targets = [s[1] for s in filtered]
        base_ds.classes = class_names

    loader = DataLoader(
        base_ds,
        batch_size=(batch_size or int(cfg["train"]["batch_size"])),
        shuffle=(split == "train"),
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True
    )
    return loader, base_ds

def load_fer2013(split, cfg, batch_size=None):
    """
    FER-2013: **solo test** en este proyecto.
    Si el YAML no define train/val para FER-2013, no fallamos: forzamos 'test'.
    """
    ds_cfg = cfg["datasets"]["fer2013"]
    # siempre usamos test_dir; si piden 'train'/'val', ignoramos y cargamos test
    ds_path = ds_cfg.get("test_dir", None)
    if ds_path is None:
        raise KeyError("[FER-2013] Se esperaba 'test_dir' en datasets.fer2013 del YAML.")

    transform = get_transforms(cfg, split="test")
    dataset = datasets.ImageFolder(root=ds_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=(batch_size or int(cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True
    )
    return loader, dataset

def load_ckplus(cfg, batch_size=None):
    """
    CK+ solo para evaluación (test). Permite excluir clases (e.g., 'contempt').
    """
    ds_path = cfg["datasets"]["ckplus"]["root_dir"]
    transform = get_transforms(cfg, split="test")

    exclude = cfg["datasets"]["ckplus"].get("exclude", [])
    valid_classes = [c for c in cfg["classes"] if c not in exclude]

    dataset = datasets.ImageFolder(root=ds_path, transform=transform)
    filtered_samples = [(p, valid_classes.index(dataset.classes[idx]))
                        for p, idx in dataset.samples if dataset.classes[idx] in valid_classes]

    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = valid_classes

    loader = DataLoader(
        dataset,
        batch_size=(batch_size or int(cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True
    )
    return loader, dataset

# -------------------------------
# Quick test
# -------------------------------
if __name__ == "__main__":
    cfg = load_config()
    # Solo fer test:
    fer_test_loader, fer_test_ds = load_fer2013("test", cfg)
    aff_train_loader, aff_train_ds = load_affectnet("train", cfg)
    aff_val_loader, aff_val_ds = load_affectnet("val", cfg)
    raf_train_loader, raf_train_ds = load_rafdb("train", cfg)
    raf_val_loader, raf_val_ds = load_rafdb("val", cfg)
    ck_loader, ck_ds = load_ckplus(cfg)

    print("[TEST] FER-2013 test batch shape:", next(iter(fer_test_loader))[0].shape)
    print("[TEST] AffectNet val batch shape:", next(iter(aff_val_loader))[0].shape)
    print("[TEST] RAF-DB train batch shape:", next(iter(raf_train_loader))[0].shape)
    print("[TEST] CK+ test batch shape:", next(iter(ck_loader))[0].shape)
