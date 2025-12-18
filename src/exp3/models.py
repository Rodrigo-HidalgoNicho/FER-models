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
    """MobileNetV3-Large preentrenada en ImageNet con capa final adaptada."""

    # Compatibilidad con versiones nuevas de TorchVision
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Reemplazar la capa final
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    # Congelar si se indica
    if freeze:
        freeze_backbone(model)
        # Aseguramos que la head sí se entrene
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Descongelar últimos bloques si aplica
    if unfreeze_blocks:
        unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)

    return model


# -------------------------------
# EfficientNet-B0
# -------------------------------
def build_efficientnet(num_classes=7, pretrained=True, freeze=False, unfreeze_blocks=None):
    """EfficientNet-B0 preentrenada en ImageNet con capa final adaptada."""

    # Compatibilidad con versiones nuevas de TorchVision
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_m(weights=weights)

    # Reemplazar la capa final
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    # Congelar si se indica
    if freeze:
        freeze_backbone(model)
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Descongelar últimos bloques si aplica
    if unfreeze_blocks:
        unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)

    return model
