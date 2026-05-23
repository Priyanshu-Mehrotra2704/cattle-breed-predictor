import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt

# Download dataset
path = kagglehub.dataset_download(
    "atharvadarpude/indian-cattle-image-dataset"
)

path = os.path.join(path, 'cattle')

print(f"Dataset path: {path}")

# Image settings
img_size = 600
batch = 16

# EfficientNet preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.1,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)

# Training data
train_data = datagen.flow_from_directory(
    path,
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode='categorical',
    subset='training'
)

print("Training Images:", train_data.samples)

# Validation data
val_data = datagen.flow_from_directory(
    path,
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode='categorical',
    subset='validation'
)

print("Validation Images:", val_data.samples)

# Load trained model
model = load_model("best_finetuned_model.keras")

# Fine-tuning
base_model = model.layers[0]

base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compile again with VERY LOW learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),

    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_finetuned_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7
)

# Fine-tune model
history = model.fit(
    train_data,
    epochs=80,
    validation_data=val_data,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# Save final model
model.save("final_finetuned_model.keras")

# Summary
model.summary()

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.show()


# =========================================================
# LOSS GRAPH
# =========================================================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.show()


# =========================================================
# TOP-3 ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(10,5))

plt.plot(history.history['top_k_categorical_accuracy'])
plt.plot(history.history['val_top_k_categorical_accuracy'])

plt.title("Top-3 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Top-3 Accuracy")

plt.legend(['Train', 'Validation'])

plt.show()