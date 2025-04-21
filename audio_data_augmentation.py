# Code written by Juliette Gelderland and Ana Antohi
# Augments the audio files to increase the dataset

import os
import librosa
import numpy as np
import soundfile as sf

input_dir = '../speech_recognition_CNN/final_audio_files'
output_dir = 'FINAL_augmented_final_audio_files'

sample_rate = 16000
duration = 1.5  # in seconds
samples = int(sample_rate * duration)

os.makedirs(output_dir, exist_ok=True)

# Augmentation functions used: time shift, pitch shift, add noise
def my_time_shift(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-n_steps, n_steps + 1))

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

# Process and save all files
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for file_name in os.listdir(class_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(class_path, file_name)

            audio, sr = librosa.load(file_path, sr=sample_rate)
            audio, _ = librosa.effects.trim(audio)

            # Resize and pad if needed
            if len(audio) < samples:
                audio = np.pad(audio, (0, samples - len(audio)), mode='constant')
            else:
                audio = audio[:samples]

            # Save original (optional, if needed in new folder)
            base_name = file_name.replace('.wav', '')
            original_out = os.path.join(output_class_path, f"{base_name}.wav")
            sf.write(original_out, audio, sample_rate)

            # Apply augmentation
            augmentations = {
                'timeshift': my_time_shift(audio),
                'pitchshift': pitch_shift(audio, sr),
                'noise': add_noise(audio)
            }

            for aug_name, aug_audio in augmentations.items():
                aug_file = os.path.join(output_class_path, f"{base_name}_{aug_name}.wav")
                sf.write(aug_file, aug_audio, sample_rate)

# For checking if the audio files were successfully saved
print("Audio augmentation complete, files are saved in:", os.path.abspath(output_dir))
