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

# Download dataset
path = kagglehub.dataset_download(
    "atharvadarpude/indian-cattle-image-dataset"
)

path = os.path.join(path, 'cattle')

print(f"Dataset path: {path}")

# Image settings
img_size = 260
batch = 16

# EfficientNet preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,

    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
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
model = load_model("final_cattle_model.h5")

# Fine-tuning
base_model = model.layers[0]

base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compile again with VERY LOW learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),

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
    epochs=30,
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