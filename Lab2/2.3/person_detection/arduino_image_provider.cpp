#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <TinyMLShield.h>
#include <algorithm>  // For std::min and std::max
#include <cmath>      // For roundf

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data,
                      float input_scale, int input_zero_point) {
  const int kCameraWidth = 176;
  const int kCameraHeight = 144;

  static bool g_is_camera_initialized = false;

  if (!g_is_camera_initialized) {
    if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7670)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true;

    Serial.begin(115200);
    while (!Serial);
  }

  uint8_t data[kCameraWidth * kCameraHeight];
  Camera.readFrame(data);

  int crop_x = (kCameraWidth - image_width) / 2;
  int crop_y = (kCameraHeight - image_height) / 2;

  // Optional: send header bytes
  Serial.write(0xAA);
  Serial.write(0x55);

  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      int src_index = (crop_y + y) * kCameraWidth + (crop_x + x);
      int dst_index = y * image_width + x;

      uint8_t pixel = data[src_index];

      // Send raw grayscale value (0–255) over Serial
      Serial.write(pixel);

      // Quantize for model
      float normalized = static_cast<float>(pixel) / 255.0f;
      int32_t quantized = static_cast<int32_t>(roundf(normalized / input_scale + input_zero_point));
      quantized = std::min<int32_t>(127, std::max<int32_t>(-128, quantized));
      image_data[dst_index] = static_cast<int8_t>(quantized);
    }
  }

  // Optional: send end bytes
  Serial.write(0x55);
  Serial.write(0xAA);

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE
