import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load ASL Dataset
data = np.load("mlkit_asl_data.npy")
labels = np.load("mlkit_asl_labels.npy")

# Normalize data (scale between 0 and 1)
data = data / np.max(data)

# Define the Model
model = keras.Sequential([
    keras.layers.Input(shape=(8,)),  # ML Kit: 4 points * (x, y)
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(len(labels), activation="softmax")  # 24 letters (A-Y, excluding J, Z)
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the Model
model.fit(data, labels, epochs=50, batch_size=32)

# Save as a TensorFlow Lite Model
model.save("asl_mlkit.h5")  # Save in HDF5 format
print("✅ ML Kit ASL Model Saved as 'asl_mlkit.tflite'")
