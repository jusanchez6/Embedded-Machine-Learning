import serial
import numpy as np
import matplotlib.pyplot as plt

# ==== Configuration ====
SERIAL_PORT = '/dev/ttyACM0'  # Replace with your port (e.g. 'COM5' on Windows or '/dev/ttyACM0' on Linux)
BAUD_RATE = 115200
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
FRAME_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
START_BYTES = bytes([0xAA, 0x55])
END_BYTES = bytes([0x55, 0xAA])

# ==== Setup Serial ====
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
print(f"Listening on {SERIAL_PORT}...")

# ==== Visualization ====
plt.ion()
fig, ax = plt.subplots()
image_display = ax.imshow(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)), cmap='gray', vmin=0, vmax=255)
plt.title("Live Camera Feed")
plt.axis('off')

def read_frame():
    while True:
        # Sync to start bytes
        if ser.read(2) == START_BYTES:
            # Read image data
            img_data = ser.read(FRAME_SIZE)
            if len(img_data) != FRAME_SIZE:
                continue  # Incomplete frame

            # Read and confirm end bytes
            if ser.read(2) == END_BYTES:
                return np.frombuffer(img_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

# ==== Main Loop ====
try:
    while True:
        frame = read_frame()
        image_display.set_data(frame)
        plt.draw()
        plt.pause(0.001)
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
