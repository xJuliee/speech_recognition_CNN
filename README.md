# speech_recognition_CNN
A Convolutional Neural Network (CNN) project for speech command recognition using MFCCs (Mel-Frequency Cepstral Coefficients). TThe model is trained to recognize five voice commands ("Forward", "Back", "Left", "Right" and "Stop") from audio input, with support for real-time inference via microphone and TensorFlow Lite deployment.

The system is designed to interface with an Arduino Mega2560 microcontroller, either standalone or with integrated WiFi (ESP8266). Real-time command recognition can be performed using mic_input_no_wifi.py (offline) or mic_input.py (WiFi-enabled). In the original application, the Arduino communicates with an RP6 Robot over the I2C protocol. Recognized voice commands are transmitted to the robot, triggering corresponding movement routines and enabling real-time voice-controlled navigation.

# Project Structure:
- audio_preprocessing.py           # Preprocessing and MFCC extraction
- audio_data_augmentation.py       # Audio augmentation functions
- speech_recognition_CNN.py        # Training the CNN and evaluation of the CNN
- mic_input.py                     # Live microphone input with WiFi
- mic_input_wth_wifi.py            # Offline microphone input
- best_model.h5                    # Best trained CNN model
- original_model.h5                # Initial model before tuning
- micro_best_model.tflite          # Lightweight TFLite model for deployment
- Pipfile                          # Python environment setup
- README.md                        # This file
- Various audio & MFCC folders     # Audio data and extracted features
- Arduino & RP6                    # Folder for Arduino & RP6 Codes and Library

# Requirements:
Key libraries:
- TensorFlow (https://www.tensorflow.org/)
- Librosa (https://librosa.org/doc/latest/index.html)
- SoundDevice (https://python-sounddevice.readthedocs.io/en/0.5.1/)
- scikit-learn (https://scikit-learn.org/stable/)
- Matplotlib, Seaborn (https://matplotlib.org/ and https://seaborn.pydata.org/)
- RP6.h library (https://github.com/b3nzchr3ur/arduino-rp6-library)

# CNN Model Architecture:
![image](https://github.com/user-attachments/assets/aefbf2b8-03fe-4849-8782-9a0a1537716c)

# Written by:
Juliette Gelderland and Ana Antohi

# Credits:
b3nzchr3ur for creating the "arduino-rp6-library" library (https://github.com/b3nzchr3ur/arduino-rp6-library)
