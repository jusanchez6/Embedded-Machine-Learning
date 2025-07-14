import numpy as np
import time
import tracemalloc
import psutil
import os
from pathlib import Path
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

# Este programa hace la prueba de inferencia del modelo de tflite 
# en la jetson nano 




# === Parámetros ===
img_size = (100, 100)
test_dir = Path("./test")
model_path = "fruit_classifier.tflite"

# === Cargar imágenes manualmente ===
X = []
y = []
class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
class_to_index = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    for img_file in (test_dir / class_name).glob("*"):
        try:
            img = Image.open(img_file).convert("RGB").resize(img_size)
            X.append(np.asarray(img))
            y.append(class_to_index[class_name])
        except Exception as e:
            print(f"Error con {img_file}: {e}")

X = np.array(X, dtype=np.float32) / 255.0  # Normalizar
y = np.array(y)

print(f"✅ Cargadas {len(X)} imágenes de prueba")

# === Cargar modelo TFLite ===
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Adaptar input según modelo ===
X = X.astype(np.float32)
if len(input_details[0]['shape']) == 4 and input_details[0]['shape'][3] == 1:
    # Grayscale
    X = np.mean(X, axis=-1, keepdims=True)  # RGB -> 1 canal

# === Medir tiempo y memoria ===
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss
tracemalloc.start()
start = time.perf_counter()

predictions = []
for img in X:
    input_tensor = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output)
    predictions.append(predicted_label)

end = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
mem_after = process.memory_info().rss

# === Evaluación ===
predictions = np.array(predictions)
accuracy = np.mean(predictions == y)

total_time = end - start
avg_inference_time = total_time / len(X) * 1000
peak_memory_MB = peak / (1024 * 1024)

JETSON_NANO_POWER_W = 5.0
energy_J = total_time * JETSON_NANO_POWER_W

# === Resultados ===
print(f"\n✅ Inferencia completada sobre {len(X)} imágenes")
print(f"⏱️  Tiempo total: {total_time:.3f} s")
print(f"⏱️  Tiempo promedio por imagen: {avg_inference_time:.3f} ms")
print(f"📈 Memoria pico: {peak_memory_MB:.2f} MB")
print(f"🎯 Precisión en test set: {accuracy * 100:.2f}%")
print(f"⚡ Energía estimada: {energy_J:.3f} J (con {JETSON_NANO_POWER_W}W)")

