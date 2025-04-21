# Code written by Juliette Gelderland and Ana Antohi
# For extracting the MFCC features from the (augmented) audio files

import os
import numpy as np
import librosa

input_base_dir = 'FINAL_augmented_final_audio_files'
output_base_dir = 'FINAL_augmented_final_data_mfcc_files'

print("Looking in:", os.path.abspath(input_base_dir))

os.makedirs(output_base_dir, exist_ok=True)

# Parameters for audio
sample_rate = 16000
n_mfcc = 40  # Standard MFCC coefficients for speech recognition
n_mel_filters = 23  # Number of Mel filters
duration = 1.5
samples_per_file = int(sample_rate * duration)
fixed_shape = (43, 40)

total_trimmed_lengths = []
num_files = 0    # Counter for the processed files

# Process samples for each directory
for class_name in os.listdir(input_base_dir):
    class_path = os.path.join(input_base_dir, class_name)
    if os.path.isdir(class_path):
        output_class_path = os.path.join(output_base_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for file_name in os.listdir(class_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_path, file_name)

                # Load audio
                signal, sr = librosa.load(file_path, sr=sample_rate)

                # Trim silence
                signal, _ = librosa.effects.trim(signal)

                #  Record the trimmed length
                total_trimmed_lengths.append(len(signal))
                num_files += 1

                # Resize the audio to 1.5s
                target_length = samples_per_file

                if len(signal) < target_length:
                    # Smaller than target length--> pad with zeros
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                else:
                    # Larger than target length --> trim to 1.5 seconds
                    signal = signal[:target_length]

                # Pre-emphasis filter
                signal = librosa.effects.preemphasis(signal, coef=0.97)

                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc,
                                            n_mels=n_mel_filters, n_fft=512, hop_length=256)

                # Add delta and delta-delta features
                delta_mfcc = librosa.feature.delta(mfcc)
                delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)

                # Stack MFCC, delta, and delta-delta features
                mfcc_combined = np.hstack([mfcc, delta_mfcc, delta_delta_mfcc])

                # Resize MFCC to fixed shape (93, 40)
                if mfcc_combined.shape[1] < fixed_shape[1]:
                    pad_width = fixed_shape[1] - mfcc_combined.shape[1]
                    mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc_combined = mfcc_combined[:, :fixed_shape[1]]

                if mfcc_combined.shape[0] < fixed_shape[0]:
                    pad_height = fixed_shape[0] - mfcc_combined.shape[0]
                    mfcc_combined = np.pad(mfcc_combined, ((0, pad_height), (0, 0)), mode='constant')
                else:
                    mfcc_combined = mfcc_combined[:fixed_shape[0], :]

                # Normalize the MFCC
                mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / np.std(mfcc_combined)

                # Save MFCC
                output_path = os.path.join(output_class_path, file_name.replace('.wav', '.npy'))
                np.save(output_path, mfcc_combined)

# Results
if num_files > 0:
    avg_trimmed_duration_sec = np.mean(total_trimmed_lengths) / sample_rate
    print(f" Average duration after silence trimming: {avg_trimmed_duration_sec:.3f} seconds")
    print(f" Average number of samples: {np.mean(total_trimmed_lengths):.0f}")
else:
    print(" No audio files were found.")

print("MFCCs with delta features saved")


