import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights, EfficientNet_V2_M_Weights

# -------------------------------
# Utils: Freezing / Unfreezing
# -------------------------------
def freeze_backbone(model):
    """Congela todas las capas del backbone excepto la head."""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_last_blocks(model, num_blocks=2):
    """
    Descongela los últimos bloques del backbone para fine-tuning.
    - MobileNetV3: desbloquea últimos N InvertedResidual.
    - EfficientNet-B0: desbloquea últimos N MBConv.
    """
    if isinstance(model, models.MobileNetV3):
        blocks = list(model.features.children())
    elif isinstance(model, models.EfficientNet):
        blocks = list(model.features.children())
    else:
        return  # fallback, no hace nada

    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

# -------------------------------
# MobileNetV3-Large
# -------------------------------
def build_mobilenetv3(num_classes=7, pretrained=True, freeze=False, unfreeze_blocks=None):
    """MobileNetV3-Large (ImageNet1K_V2), head 7 clases, Dropout 0.4."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Mantener estructura original y ajustar solo dropout final y la capa FC
    # classifier = [0] Linear(960,1280), [1] Hardswish, [2] Dropout(p=0.2), [3] Linear(1280,1000)
    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(0.4)                 # subir dropout a 0.4
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if freeze:
        freeze_backbone(model)
        for p in model.classifier.parameters():
            p.requires_grad = True

    if unfreeze_blocks:
        unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)

    return model


# -------------------------------
# EfficientNet-B0
# -------------------------------
def build_efficientnetv2_m(num_classes=7, pretrained=True, freeze=False, unfreeze_blocks=None):
    """EfficientNet-V2-M (ImageNet1K_V1), head 7 clases, Dropout 0.4."""
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_m(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(0.4)
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze:
        freeze_backbone(model)
        for p in model.classifier.parameters():
            p.requires_grad = True

    if unfreeze_blocks:
        unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)

    return model
