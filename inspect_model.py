import json
import h5py
from tensorflow import keras

PATH = "face_model.h5"

def read_model_config(attrs):
    cfg_raw = attrs.get("model_config", None)
    if cfg_raw is None:
        return None
    if isinstance(cfg_raw, (bytes, bytearray)):
        cfg_raw = cfg_raw.decode("utf-8")
    # a veces ya viene como str (tu caso)
    if isinstance(cfg_raw, str):
        return json.loads(cfg_raw)
    # fallback raro
    return json.loads(str(cfg_raw))

with h5py.File(PATH, "r") as f:
    print("Keys:", list(f.keys()))  # ['model_weights', 'optimizer_weights'] es normal
    print("Attrs:", list(f.attrs.keys()))  # ['backend','keras_version','model_config','training_config']

    cfg = read_model_config(f.attrs)
    if cfg:
        print("-> Contiene arquitectura:", cfg.get("class_name", "desconocida"))
    else:
        print("-> No pude leer la arquitectura (podría ser solo pesos).")

print("\n== Intentando load_model ==")
try:
    model = keras.models.load_model(PATH, compile=True)  # intenta con compile=True
    print("Modelo cargado OK.")
    model.summary()
    print("\nInput shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    # intenta detectar activación y #clases
    try:
        last = model.layers[-1]
        act = getattr(last, "activation", None)
        units = getattr(last, "units", None)
        print("Capa final:", last.name, "| Activación:", getattr(act, "__name__", act))
        print("Unidades (posibles clases):", units)
    except Exception:
        pass

    # ¿estaba compilado?
    print("Loss configurado:", model.loss)
    print("Metrics:", model.metrics_names)
except Exception as e:
    print("No pude cargar con load_model():", repr(e))
    print("Si menciona 'Unknown layer' o 'deserialize', hace falta custom_objects o ajustar versión TF/Keras.")
