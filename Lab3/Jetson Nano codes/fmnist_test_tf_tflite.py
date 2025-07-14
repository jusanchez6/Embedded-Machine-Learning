import tensorflow as tf
import numpy as np
import tracemalloc
import time
from tensorflow.lite.python.interpreter import Interpreter  # Usamos TF directamente

# === Cargar dataset Fashion MNIST (solo test) ===
fashion_mnist = tf.keras.datasets.fashion_mnist
_, (test_images, test_labels) = fashion_mnist.load_data()
test_images = test_images / 255.0  # Normalizar
test_images = test_images.astype(np.float32)
input_image = np.expand_dims(test_images[0], axis=0)



# === Cargar modelo TFLite ===
interpreter = Interpreter(model_path='fmnist.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Measure latency and memory ---
n_runs = 1000
tracemalloc.start()
start = time.perf_counter()

for _ in range(n_runs):
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])

end = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

avg_latency_ms = (end - start) / n_runs * 1000
peak_memory_mb = peak / (1024 * 1024)

# --- Get prediction for reporting ---
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)


print(f"\n \n ============================ Resultados ================================")

print(f"\nTFLite Inference Results")
print(f"  Predicted class: {predicted_class}")
print(f"  Average Latency: {avg_latency_ms:.2f} ms")
print(f"  Peak Memory Usage: {peak_memory_mb:.4f} MB")

