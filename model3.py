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
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("✅ GPU ENABLED")
        print("GPU:", gpus)

    except RuntimeError as e:
        print(e)
else:
    print("❌ GPU NOT DETECTED")


# Download dataset
path = kagglehub.dataset_download(
    "atharvadarpude/indian-cattle-image-dataset"
)

path = os.path.join(path, 'cattle')

print(f"Dataset path: {path}")

# Image settings
img_size = 300
batch = 32

# Data augmentation
datagen = ImageDataGenerator(
    validation_split=0.1,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,

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
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(train_data.num_classes, activation='softmax')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.00005)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
# Compile model
model.compile(
    loss=loss,
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




