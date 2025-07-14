import tensorflow as tf
import numpy as np
import os
import time
import tracemalloc
import psutil
from PIL import Image
from pathlib import Path
from tensorflow.keras import layers, models

# === Parámetros ===
IMG_HEIGHT, IMG_WIDTH = 100, 100
img_size = (IMG_HEIGHT, IMG_WIDTH)
test_dir = Path("./test")

# === Obtener clases ===
class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
class_to_index = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)

# === Definir arquitectura del modelo ===
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='softmax')
])

# === Cargar pesos entrenados ===
model.load_weights("fruit_classifier.h5")

# === Cargar imágenes manualmente ===
X = []
y = []

for class_name in class_names:
    class_path = test_dir / class_name
    for img_file in class_path.glob("*"):
        try:
            img = Image.open(img_file).convert("RGB").resize(img_size)
            X.append(np.asarray(img))
            y.append(class_to_index[class_name])
        except Exception as e:
            print(f"Error al cargar {img_file}: {e}")

X = np.array(X, dtype=np.float32) / 255.0  # Normalizar manualmente
y = np.array(y)

print(f"✅ Cargadas {len(X)} imágenes de prueba")

# === Medir tiempo y memoria ===
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss
tracemalloc.start()
start_time = time.perf_counter()

# === Inferencia ===
predictions = model.predict(X)

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
mem_after = process.memory_info().rss

# === Métricas ===
total_time = end_time - start_time
avg_inference_time = total_time / len(X) * 1000  # ms por imagen
peak_memory_MB = (mem_after - mem_before) / (1024 * 1024)

# === Precisión (opcional) ===
pred_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(pred_labels == y)

# === Estimación de energía ===
JETSON_NANO_POWER_W = 5.0
energy_J = total_time * JETSON_NANO_POWER_W

# === Resultados ===

print(f"\n \n \n ============================ Resultados =======================")

print(f"\nInferencia completada sobre {len(X)} imágenes")
print(f"Tiempo total: {total_time:.3f} s")
print(f"Tiempo promedio por imagen: {avg_inference_time:.3f} ms")
print(f"Memoria pico: {peak_memory_MB:.2f} MB")
print(f"Precisión: {accuracy*100:.2f}%")
print(f"Energía estimada: {energy_J:.3f} J (con {JETSON_NANO_POWER_W}W)")
