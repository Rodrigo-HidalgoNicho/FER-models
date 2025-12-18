import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------
# Config loader
# -------------------------------
def load_config(path="configs/phase3M.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------
# Compute class weights
# -------------------------------
def compute_class_weights(dataset, num_classes):
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=num_classes)
    weights = 1.0 / (counts.float() + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights

# -------------------------------
# MixUp / CutMix utils
# -------------------------------
def mixup_data(x, y, alpha=0.2):
    """Aplica MixUp a un batch de imágenes"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Aplica CutMix a un batch de imágenes"""
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)

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
# Common transforms (dependen de fase)
# -------------------------------
def get_transforms(cfg, split="train", phase="phase1"):
    img_size = tuple(cfg["phases"][phase]["input_size"][:2])

    pil_transforms, tensor_transforms = [], []

    if "randaugment" in cfg["phases"][phase]["augmentations"]:
        pil_transforms.append(transforms.RandAugment(num_ops=2, magnitude=9))
    if "flip" in cfg["phases"][phase]["augmentations"]:
        pil_transforms.append(transforms.RandomHorizontalFlip())
    if "randomcrop" in cfg["phases"][phase]["augmentations"]:
        pil_transforms.append(transforms.RandomResizedCrop(img_size))

    pil_transforms.extend([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size)
    ])

    tensor_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if "randomerasing" in cfg["phases"][phase]["augmentations"]:
        tensor_transforms.append(transforms.RandomErasing(p=0.25))

    return transforms.Compose(pil_transforms + tensor_transforms)

# -------------------------------
# Helper para DataLoader
# -------------------------------
def build_loader(dataset, cfg, split):
    num_workers = cfg["train"].get("num_workers", 0)  # configurable desde YAML
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

# -------------------------------
# Loaders
# -------------------------------
def load_fer2013(split, cfg, batch_size=None, phase="phase2"):
    ds_path = cfg["datasets"]["fer2013"][f"{split}_dir"]
    transform = get_transforms(cfg, split, phase)
    dataset = datasets.ImageFolder(root=ds_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True
    )
    return loader, compute_class_weights(dataset, len(cfg["classes"]))

def load_affectnet(split, cfg, batch_size=None, phase="phase1"):
    ds_path = cfg["datasets"]["affectnet"][f"{split}_dir"]
    transform = get_transforms(cfg, split, phase)
    dataset = datasets.ImageFolder(root=ds_path, transform=transform)

    mapping, class_names = cfg["affectnet_map"], cfg["classes"]
    filtered_samples = [(p, class_names.index(mapping[str(l)]))
                        for p, l in dataset.samples if mapping[str(l)] is not None]

    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = class_names

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True
    )
    return loader, compute_class_weights(dataset, len(cfg["classes"]))

def load_rafdb(split, cfg, batch_size=None, phase="phase1"):
    ds_path = cfg["datasets"]["rafdb"][f"{split}_dir"]
    transform = get_transforms(cfg, split, phase)
    dataset = datasets.ImageFolder(root=ds_path, transform=transform)

    mapping, class_names = cfg["rafdb_map"], cfg["classes"]
    filtered_samples = [(p, class_names.index(mapping[str(dataset.classes[l])]))
                        for p, l in dataset.samples]

    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = class_names

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True
    )
    return loader, compute_class_weights(dataset, len(cfg["classes"]))

def load_ckplus(cfg, batch_size=None, phase="phase2"):
    ds_path = cfg["datasets"]["ckplus"]["root_dir"]
    transform = get_transforms(cfg, split="test", phase=phase)

    exclude = cfg["datasets"]["ckplus"].get("exclude", [])
    valid_classes = [c for c in cfg["classes"] if c not in exclude]

    dataset = datasets.ImageFolder(root=ds_path, transform=transform)
    filtered_samples = [(p, valid_classes.index(dataset.classes[l]))
                        for p, l in dataset.samples if dataset.classes[l] in valid_classes]

    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = valid_classes

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True
    )
    return loader, compute_class_weights(dataset, len(valid_classes))

# -------------------------------
# Quick test
# -------------------------------
if __name__ == "__main__":
    cfg = load_config()
    fer_train, fer_w = load_fer2013("train", cfg, phase="phase2")
    aff_val, aff_w = load_affectnet("val", cfg, phase="phase1")
    raf_train, raf_w = load_rafdb("train", cfg, phase="phase1")
    ck, ck_w = load_ckplus(cfg, phase="phase2")

    print("[TEST] FER-2013:", next(iter(fer_train))[0].shape, fer_w)
    print("[TEST] AffectNet:", next(iter(aff_val))[0].shape, aff_w)
    print("[TEST] RAF-DB:", next(iter(raf_train))[0].shape, raf_w)
    print("[TEST] CK+:", next(iter(ck))[0].shape, ck_w)
