/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "detection_responder.h"

#include "Arduino.h"

// Flash the blue LED after each inference
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t paper_score, int8_t rock_score, int8_t scissors_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);


  // Switch on the RED LED if paper is detected,
  // GREEN for rock, and BLUE for scissors.
  if (paper_score > rock_score && paper_score > scissors_score) {
    digitalWrite(LEDR, LOW);  // Paper detected
    TF_LITE_REPORT_ERROR(error_reporter, "Paper detected: %d", paper_score);

  } else if (rock_score > paper_score && rock_score > scissors_score) {
    digitalWrite(LEDG, LOW);  // Rock detected
    TF_LITE_REPORT_ERROR(error_reporter, "Rock detected: %d", rock_score);

  } else if (scissors_score > paper_score && scissors_score > rock_score) {
    digitalWrite(LEDB, LOW);  // Scissors detected
    TF_LITE_REPORT_ERROR(error_reporter, "Scissors detected: %d", scissors_score);

  }
  

  // TF_LITE_REPORT_ERROR(error_reporter, "Paper: %d, Rock: %d, Scissors: %d", paper_score, rock_score, scissors_score);
}

#endif  // ARDUINO_EXCLUDE_CODE
