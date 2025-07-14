import tensorflow as tf
import numpy as np
import tracemalloc
import time

# === Cargar modelo y pesos ===
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


with tf.device('/CPU:0'):

    model = models.Sequential([
	    layers.Flatten(input_shape=(28, 28)),
	    layers.Dense(128, activation='relu'),
	    layers.Dense(10, activation='softmax')
    ])

    model.load_weights("fmnist.h5")



    # === Cargar dataset Fashion MNIST (solo test) ===
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (_, _), (test_images, test_labels) = fashion_mnist.load_data()

    # Preprocesar imágenes
    test_images = test_images.astype(np.float32) / 255.0  # Normalizar

    # === Iniciar medición de tiempo y memoria ===
    tracemalloc.start()
    start_time = time.perf_counter()

    # Ejecutar predicciones (esto puede consumir bastante RAM si se hace sobre todas las imágenes a la vez)
    predictions = model.predict(test_images, batch_size=32)  # Puedes ajustar el batch_size para ahorrar RAM

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # === Métricas ===
    latency_ms = (end_time - start_time) * 1000
    peak_memory_mb = peak / (1024 * 1024)

    # === Calcular precisión ===
    accuracy = np.mean(np.argmax(predictions, axis=1) == test_labels)

    # === Mostrar resultados ===
    print(f"Latencia total: {latency_ms:.2f} ms")
    print(f"Memoria pico: {peak_memory_mb:.2f} MB")
    print(f"Precisión: {accuracy*100:.2f}%")


