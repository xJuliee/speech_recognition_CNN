# speech_recognition_CNN
A Convolutional Neural Network (CNN) project for speech command recognition using MFCCs (Mel-Frequency Cepstral Coefficients). This model is trained to recognize specific voice commands ("Forward", "Back", "Left", "Right" and "Stop") from audio input, with support for real-time inference via microphone and TensorFlow Lite deployment.

# Project Structure:
.
├── Audio.py                         # Preprocessing and MFCC extraction
├── data_augmentation.py            # Audio augmentation functions
├── main.py                         # Training and evaluation
├── mic_input.py                    # Live microphone input with WiFi
├── mic_input_no_wifi.py           # Offline microphone input
├── best_model.h5                   # Best trained CNN model
├── original_model.h5              # Initial model before tuning
├── tflite_model.tflite             # Lightweight TFLite model for deployment
├── Pipfile                         # Python environment setup
├── README.md                       # This file
├── [Various audio & MFCC folders]  # Audio data and extracted features

# Requirements:
Key libraries:
- TensorFlow (https://www.tensorflow.org/)
- Librosa (https://librosa.org/doc/latest/index.html)
- SoundDevice (https://python-sounddevice.readthedocs.io/en/0.5.1/)
- scikit-learn (https://scikit-learn.org/stable/)
- Matplotlib, Seaborn (for visualization) (https://matplotlib.org/ and https://seaborn.pydata.org/)

# Written by:
Juliette Gelderland and Ana Antohi
