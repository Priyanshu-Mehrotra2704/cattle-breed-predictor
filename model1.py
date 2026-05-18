import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D
)
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
img_size = 300
batch = 40

# Data augmentation
datagen = ImageDataGenerator(
    validation_split=0.3,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,

    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
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

# Visualize augmented images
# images, labels = next(train_data)

# for i in range(5):
#     plt.imshow(images[i])
#     plt.axis("off")
#     plt.show()

# CNN Model
base_model = EfficientNetB2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)
base_model.trainable = False
model = Sequential([
    base_model,
    
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

# Callbacks

# Stops training if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Saves best model automatically
checkpoint = ModelCheckpoint(
    "best_cattle_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Reduce learning rate automatically
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.000001
)

# Train model
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

# Model summary
model.summary()

# Save final model
model.save("final_cattle_model.h5")




