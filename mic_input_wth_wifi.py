# Code written by Juliette Gelderland and Ana Antohi
# Labels microphone input based on the pre-trained CNN model
# Sends the predicted labels to ATmega2560 via UART Serial Communication (USB)

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import serial

# --- Load Pretrained Model ---
model = load_model('../speech_recognition_CNN/best_model.h5')

# --- Load Label Encoder ---
le = LabelEncoder()
le.classes_ = np.array(['Back', 'Forward', 'Left', 'Right', 'Silence', 'Stop', 'Unknown'])

# --- Parameters ---
sample_rate = 16000
duration = 1.5
fixed_shape = (43, 40)
samples_per_file = int(sample_rate * duration)
silence_threshold = 0.01
prediction_cooldown = 0.2

# --- Setup Serial Communication ---
arduino = serial.Serial('COM13', 9600)  # Change 'COM3' to your Arduino port

# --- Preprocessing ---
def preprocess_mfcc(mfcc):
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

# --- Audio Callback ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)

    rms = np.sqrt(np.mean(np.square(indata)))
    print(f"Microphone RMS: {rms:.6f}", flush=True)

    if rms < silence_threshold:
        predicted_class = 4  # 'Silence'
        predicted_label = le.inverse_transform([predicted_class])[0]
        print(f"Predicted class: {predicted_label} (Silence) RMS: {rms:.6f}")
        time.sleep(prediction_cooldown)
    else:
        signal = indata.flatten()
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40, n_fft=512, hop_length=256)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
        mfcc_combined = np.hstack([mfcc, delta_mfcc, delta_delta_mfcc])

        if mfcc_combined.shape[1] < fixed_shape[1]:
            mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, fixed_shape[1] - mfcc_combined.shape[1])), mode='constant')
        else:
            mfcc_combined = mfcc_combined[:, :fixed_shape[1]]

        if mfcc_combined.shape[0] < fixed_shape[0]:
            mfcc_combined = np.pad(mfcc_combined, ((0, fixed_shape[0] - mfcc_combined.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc_combined = mfcc_combined[:fixed_shape[0], :]

        mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / np.std(mfcc_combined)
        processed_mfcc = preprocess_mfcc(mfcc_combined)

        predictions = model.predict(processed_mfcc)
        predicted_class = np.argmax(predictions)
        predicted_prob = np.max(predictions)

        confidence_threshold = 0.3
        if predicted_prob < confidence_threshold:
            predicted_class = 6  # 'Unknown'
        predicted_label = le.inverse_transform([predicted_class])[0]

        print(f"Predicted class: {predicted_label} (Index: {predicted_class})")
        print("Prediction probabilities:", ['{:.4f}'.format(p) for p in predictions[0]])

        if predicted_label not in ["Silence", "Unknown"]:
            try:
                arduino.write((predicted_label.lower() + '\n').encode())  # Convert to lowercase
                print("Sent to Arduino:", predicted_label)
                # Add cooldown to prevent fast consecutive predictions
                time.sleep(prediction_cooldown)
            except Exception as e:
                print("Failed to send to Arduino:", e)
            time.sleep(0.1)

    time.sleep(0.2)


# --- Start Microphone Stream ---
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=samples_per_file // 2):
    print("Recording audio... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
