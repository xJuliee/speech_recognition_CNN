# Code written by Juliette Gelderland and Ana Antohi
# Building, training and saving a CNN on audio inputs
# The dataset used is a combination of our own collected data and Google Speech Commands dataset - Kaggle. https://www.kaggle.com/datasets/neehakurelli/google-speech-commands

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Load Data
data_path = 'collected_and_google_mfcc_files'
fixed_shape = (43, 40)   # Same fixed shaped used when extracting MFCC features

X, y = [], []

for label in os.listdir(data_path):
    label_dir = os.path.join(data_path, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        if file.endswith('.npy'):
            mfcc = np.load(os.path.join(label_dir, file))

            # Ensure the MFCC shape, if different tha =n fixed shape pad/trim
            if mfcc.shape[0] < fixed_shape[0]:
                pad_height = fixed_shape[0] - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_height), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:fixed_shape[0], :]

            if mfcc.shape[1] < fixed_shape[1]:
                pad_width = fixed_shape[1] - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :fixed_shape[1]]

            X.append(mfcc)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Normalize
X = (X - np.mean(X)) / np.std(X)
X = X[..., np.newaxis]

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
n_classes = len(np.unique(y_encoded))

# Check the label-encoded values
print("Label Encoded Values (y_encoded):")
print(y_encoded)

print("Original classes (labels):")
print(le.classes_)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Hyperparameters for Grid Search
param_grid = {
    'batch_size': [16, 32],
    'epochs': [10, 20],
    'dropout_rate': [0.2,0.5],
    'filters': [32, 64],
    'dense_units': [64, 128],
    'l2_regularization': [0.001]  # For model weight penalty --> prevents overfitting
}

# For best model
best_score = 0
best_model = None
best_params = {}
best_history = None

# Perform manual grid search
for batch_size, epochs, dropout_rate, filters, dense_units, l2_reg in itertools.product(
    param_grid['batch_size'],
    param_grid['epochs'],
    param_grid['dropout_rate'],
    param_grid['filters'],
    param_grid['dense_units'],
    param_grid['l2_regularization']
):
    print(f"\nTraining with batch_size={batch_size}, epochs={epochs}, "
          f"dropout={dropout_rate}, filters={filters}, dense_units={dense_units}, l2={l2_reg}")

    # Build model
    model = Sequential([
        Conv2D(filters, (3, 3), activation='relu', input_shape=X_train.shape[1:],
               kernel_regularizer=l2(l2_reg)),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Conv2D(filters * 2, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Conv2D(filters * 4, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),  # Added layer
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Flatten(),
        Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.1),

        Dense(n_classes, activation='softmax')
    ])

    # Check input shape
    model.summary()
    print("Input shape:", model.input_shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping if teh accuracy doesn't improve
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stopping]  # Early stopping
    )

    val_acc = history.history['val_accuracy'][-1]

    # Save best model
    if val_acc > best_score:
        best_score = val_acc
        best_model = model
        best_params = {
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout_rate': dropout_rate,
            'filters': filters,
            'dense_units': dense_units,
            'l2_regularization': l2_reg
        }
        best_history = history

print("\n Best Parameters:")
print(best_params)
print(f"Best Validation Accuracy: {best_score:.4f}")

# Plot training history to check overfitting/underfitting
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'], label='Train Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Best Model Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Best Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate Best Model
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {test_acc:.4f}")

# Predictions
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Report for best model
print("\n Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the best model to an HDF5 or Keras file
model_path = "../MOD7Project - CNN with manual search/best_model.h5"
best_model.save(model_path)
print(f"\n Best model is saved as: {model_path}")

# Save the original model as .h5
h5_model_path = '../MOD7Project - CNN with manual search/original_model.h5'
model.save(h5_model_path)
print(f"Original model is saved as {h5_model_path}")

#
# # Convert the model to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# # # Apply post-training quantization to convert to INT8
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # Set input type to INT8
# converter.inference_output_type = tf.int8  # Set output type to INT8
#
# # Use representative dataset for quantization (needed for INT8)
# def representative_data_gen():
#     for input_value in X_train[:100]:  # Use a subset of training data for quantization
#         # Add an extra dimension to match the batch size (1 in this case)
#         input_value = np.expand_dims(input_value, axis=0)  # Now shape is (1, height, width, channels)
#         yield [input_value.astype(np.float32)]
#
# converter.representative_dataset = representative_data_gen
#
# # Convert the model
# tflite_model = converter.convert()
#
# # Save the quantized model to a .tflite file
# # tflite_model_path = '../MOD7Project - CNN with manual search/tflite_model.tflite'
# # with open(tflite_model_path, 'wb') as f:
# #     f.write(tflite_model)
#
# print(f"Model has been converted and saved as {tflite_model_path}")
