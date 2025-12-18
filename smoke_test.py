import numpy as np
from tensorflow import keras

m = keras.models.load_model("face_model.h5", compile=False)
print("Input shape:", m.input_shape, "Output shape:", m.output_shape)

# crea batch dummy con la forma correcta
inp = list(m.input_shape)
inp[0] = 2  # batch 2
# si hay None en alguna dim, cámbiala por un valor concreto (48, 224, etc.)
inp = [d if d is not None else 48 for d in inp]
x = np.random.rand(*inp).astype("float32")

y = m.predict(x)
print("Pred shape:", y.shape)
print("Pred[0]:", y[0])
