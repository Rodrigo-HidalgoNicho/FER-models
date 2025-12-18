import sys, cv2, numpy as np
from tensorflow import keras

if len(sys.argv) < 2:
    print("Uso: python infer_image.py ruta/imagen.jpg")
    sys.exit(1)

# Cargar modelo
model = keras.models.load_model("face_model.h5")

# Parametros del input
H, W, C = 48, 48, 1

# Cargar imagen
img = cv2.imread(sys.argv[1])
if img is None:
    raise FileNotFoundError("No pude abrir la imagen.")

# Preprocesar a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (W, H))

# Normalizar y dar forma (1,H,W,1)
x = resized.astype("float32") / 255.0
x = np.expand_dims(x, axis=(0, -1))

# Predecir
pred = model.predict(x)
print("Probabilidades:", pred[0])
print("Clase predicha:", np.argmax(pred[0]))
