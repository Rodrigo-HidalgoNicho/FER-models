import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_curves(csv_path, output_dir="outputs/plots"):
    # Cargar historial
    history = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # --- 2. Accuracy ---
    if "accuracy" in history.columns and "val_accuracy" in history.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(history["accuracy"], label="train_acc")
        plt.plot(history["val_accuracy"], label="val_acc")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
        plt.close()

    # --- 3. F1 Macro ---
    if "val_f1_macro" in history.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(history["val_f1_macro"], label="val_f1_macro", color="purple")
        plt.title("Validation F1 Macro")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Macro")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "f1_macro_curve.png"))
        plt.close()

    print(f"[INFO] Gráficas guardadas en {output_dir}")


if __name__ == "__main__":
    csv_path = "outputs/csv/train_history.csv"  # ajusta si guardas en otra ruta
    plot_curves(csv_path)
