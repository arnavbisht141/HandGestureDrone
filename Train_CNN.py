import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

import os
import shutil
import random

SOURCE_DIR = "dataset/train"
TARGET_DIR = "dataset/val"
SPLIT = 0.2

os.makedirs(TARGET_DIR, exist_ok=True)

for cls in os.listdir(SOURCE_DIR):
    src_cls = os.path.join(SOURCE_DIR, cls)
    dst_cls = os.path.join(TARGET_DIR, cls)
    os.makedirs(dst_cls, exist_ok=True)

    images = os.listdir(src_cls)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT)

    for img in images[:split_idx]:
        shutil.move(
            os.path.join(src_cls, img),
            os.path.join(dst_cls, img)
        )

# ===================== CONFIG =====================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
SEED = 42

train_ds = image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_ds = image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)


# ===================== LOAD DATA =====================
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ===================== NORMALIZATION =====================
normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ===================== CNN MODEL =====================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu",
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.summary()

# ===================== COMPILE =====================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ===================== TRAIN =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===================== SAVE =====================
model.save("gesture_cnn_baseline.h5")
print("Model saved as gesture_cnn_baseline.h5")
