import os
import torch
import torch.onnx
import shutil
from dataloaders import load_config
from models import build_mobilenetv3, build_efficientnet


def export_to_onnx(backbone="mobilenetv3"):
    # Cargar configuración específica del experimento 3
    cfg = load_config("configs/Phase3E3.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"]["num_classes"]

    # Directorios específicos de EXP3
    if backbone == "mobilenetv3":
        ckpt_dir = "outputs/phase3/mobilenetv3/exp3/checkpoints"
        export_dir = "outputs/phase3/mobilenetv3/exp3/exports"
    elif backbone == "efficientnetb0":
        ckpt_dir = "outputs/phase3/efficientnetb0/exp3/checkpoints"
        export_dir = "outputs/phase3/efficientnetb0/exp3/exports"
    else:
        raise ValueError(f"Backbone desconocido: {backbone}")

    os.makedirs(export_dir, exist_ok=True)

    # Ruta del mejor checkpoint
    best_ckpt = os.path.join(ckpt_dir, f"best_{backbone}.pt")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"No se encontró el checkpoint: {best_ckpt}")

    # Construir modelo según backbone
    if backbone == "mobilenetv3":
        model = build_mobilenetv3(num_classes=num_classes, pretrained=False)
    elif backbone == "efficientnetb0":
        model = build_efficientnet(num_classes=num_classes, pretrained=False)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # Dummy input para exportación ONNX
    input_h, input_w = cfg["model"]["input_size"][:2]
    dummy_input = torch.randn(1, 3, input_h, input_w).to(device)
    onnx_path = os.path.join(export_dir, f"model_{backbone}.onnx")

    # Exportar modelo a formato ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"[INFO] Modelo exportado a ONNX en {onnx_path}")

    # Comprimir resultados (modelos + artefactos)
    parent_dir = os.path.dirname(export_dir)  # -> outputs/phase3/mobilenetv3/exp3
    zip_path = os.path.join(parent_dir, f"phase3_train_{backbone}")
    shutil.make_archive(zip_path, 'zip', export_dir)

    print(f"[INFO] Resultados comprimidos en {zip_path}.zip")


if __name__ == "__main__":
    # Puedes cambiar el backbone según lo que desees exportar
    export_to_onnx("mobilenetv3")
    # export_to_onnx("efficientnetb0")
