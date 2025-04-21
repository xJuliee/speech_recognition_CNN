# Code written by Juliette Gelderland and Ana Antohi
# Labels microphone input based on the pre-trained CNN model
# Sends the predicted labels to ESP8266 via Wi-Fi

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import socket

# Setup the TCP Connection
esp8266_ip = "192.168.54.75"  # Replace with the IP printed by the ESP8266
esp8266_port = 5000

# Create TCP socket to connect to the ESP8266
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((esp8266_ip, esp8266_port))
print(f"Connected to ESP8266 at {esp8266_ip}:{esp8266_port}")

# Load the saved pre-trained model
model = load_model('../MOD7Project - CNN with manual search/best_model.h5')

# Label Encoder
le = LabelEncoder()
le.classes_ = np.array(['Back', 'Forward', 'Left', 'Right', 'Silence', 'Stop', 'Unknown'])

# Parameters for the audio
sample_rate = 16000
duration = 1.5
fixed_shape = (43, 40)
samples_per_file = int(sample_rate * duration)
silence_threshold = 0.01
prediction_cooldown = 0.2

# Preprocessing the mfcc files
def preprocess_mfcc(mfcc):
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

# Audio Callback
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)

    rms = np.sqrt(np.mean(np.square(indata)))
    # Print RMS to check for silence
    print(f"Microphone RMS: {rms:.6f}", flush=True)

    if rms < silence_threshold:
        # Audio input value is under threshold --> treat it as silence
        predicted_class = 4  # Index for 'Silence'
        predicted_label = le.inverse_transform([predicted_class])[0]
        print(f"Predicted class: {predicted_label} (Silence) RMS: {rms:.6f}")
        time.sleep(prediction_cooldown)
    else:
        # Extract mfcc
        signal = indata.flatten()
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40, n_fft=512, hop_length=256)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
        mfcc_combined = np.hstack([mfcc, delta_mfcc, delta_delta_mfcc])

        # PAdd mfcc to fixed shape
        if mfcc_combined.shape[1] < fixed_shape[1]:
            mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, fixed_shape[1] - mfcc_combined.shape[1])), mode='constant')
        else:
            mfcc_combined = mfcc_combined[:, :fixed_shape[1]]

        if mfcc_combined.shape[0] < fixed_shape[0]:
            mfcc_combined = np.pad(mfcc_combined, ((0, fixed_shape[0] - mfcc_combined.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc_combined = mfcc_combined[:fixed_shape[0], :]

        mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / np.std(mfcc_combined)

        # Input for the model
        mfcc_for_model = preprocess_mfcc(mfcc_combined)

        predictions = model.predict(mfcc_for_model)
        predicted_class = np.argmax(predictions)
        predicted_prob = np.max(predictions)

        confidence_threshold = 0.3
        if predicted_prob < confidence_threshold:
            predicted_class = 6  # Index for 'Unknown'
        predicted_label = le.inverse_transform([predicted_class])[0]

        print(f"Predicted class: {predicted_label} (Index: {predicted_class})")
        print("Prediction probabilities:", ['{:.4f}'.format(p) for p in predictions[0]])

        # Send commands to teh ESP8266
        if predicted_label not in ["Silence", "Unknown"]:
            try:
                message = predicted_label.lower() + '\n'
                sock.send(message.encode())  # Send to ESP8266 via Wi-Fi
                print("Sent to ESP8266:", predicted_label)
                time.sleep(prediction_cooldown)
            except Exception as e:
                print("Failed to send to ESP8266:", e)
            time.sleep(0.1)

    time.sleep(0.2)


# Start Microphone Stream --> microphone continuously listens
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=samples_per_file // 2):
    print("Recording audio... Press Ctrl+C to stop.")
    while True:
     time.sleep(1)
