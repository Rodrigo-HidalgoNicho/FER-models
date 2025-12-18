import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------
# Config loader
# -------------------------------
def load_config(path="/kaggle/working/modelFER/configs/phase3.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Si no existe "classes", lo construimos automáticamente
    if "classes" not in cfg:
        # Recolectar nombres de clases válidos desde los mappings
        class_names = set()

        if "affectnet_map" in cfg:
            class_names.update([v for v in cfg["affectnet_map"].values() if v])
        if "rafdb_map" in cfg:
            class_names.update([v for v in cfg["rafdb_map"].values() if v])
        if "ckplus_map" in cfg:
            class_names.update([v for v in cfg["ckplus_map"].values() if v])

        cfg["classes"] = sorted(list(class_names))

    return cfg

# -------------------------------
# Compute class weights
# -------------------------------
def compute_class_weights(dataset, num_classes):
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=num_classes)
    weights = 1.0 / (counts.float() + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights

# -------------------------------
# Common transforms
# -------------------------------
def get_transforms(cfg, split="train"):
    img_size = tuple(cfg["model"]["input_size"][:2])
    if split == "train":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# -------------------------------
# Helper para DataLoader
# -------------------------------
def build_loader(dataset, cfg, split):
    num_workers = cfg["train"].get("num_workers", 0)
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
def load_fer2013(split, cfg, batch_size=None):
    ds_path = cfg["datasets"]["fer2013"][f"{split}_dir"]
    transform = get_transforms(cfg, split)
    dataset = datasets.ImageFolder(root=ds_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=True
    )
    return loader, compute_class_weights(dataset, len(cfg["classes"]))

def load_affectnet(split, cfg, batch_size=None):
    ds_path = cfg["datasets"]["affectnet"][f"{split}_dir"]
    transform = get_transforms(cfg, split)
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

def load_rafdb(split, cfg, batch_size=None):
    ds_path = cfg["datasets"]["rafdb"][f"{split}_dir"]
    transform = get_transforms(cfg, split)
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

def load_ckplus(cfg, batch_size=None):
    ds_path = cfg["datasets"]["ckplus"]["root_dir"]
    transform = get_transforms(cfg, split="test")

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
    fer_train, fer_w = load_fer2013("train", cfg)
    aff_val, aff_w = load_affectnet("val", cfg)
    raf_train, raf_w = load_rafdb("train", cfg)
    ck, ck_w = load_ckplus(cfg)

    print("[TEST] FER-2013:", next(iter(fer_train))[0].shape, fer_w)
    print("[TEST] AffectNet:", next(iter(aff_val))[0].shape, aff_w)
    print("[TEST] RAF-DB:", next(iter(raf_train))[0].shape, raf_w)
    print("[TEST] CK+:", next(iter(ck))[0].shape, ck_w)
