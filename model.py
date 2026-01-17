import os
import random
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================
# PATHS (MATCHING YOUR FOLDER STRUCTURE)
# =====================================
SOURCE_AI = "AI-face-detection-Dataset/AI"
SOURCE_REAL = "AI-face-detection-Dataset/real"

BASE_DIR = "final_dataset"
TRAIN_AI = os.path.join(BASE_DIR, "train/AI")
TRAIN_REAL = os.path.join(BASE_DIR, "train/REAL")
TEST_AI = os.path.join(BASE_DIR, "test/AI")
TEST_REAL = os.path.join(BASE_DIR, "test/REAL")

# =====================================
# CREATE REQUIRED FOLDERS
# =====================================
for folder in [TRAIN_AI, TRAIN_REAL, TEST_AI, TEST_REAL]:
    os.makedirs(folder, exist_ok=True)

# =====================================
# SPLIT DATASET (80% TRAIN / 20% TEST)
# =====================================
def split_and_copy(src, train_dest, test_dest, split_ratio=0.8):
    files = os.listdir(src)
    random.shuffle(files)

    split = int(len(files) * split_ratio)
    train_files = files[:split]
    test_files = files[split:]

    for f in train_files:
        shutil.copy(os.path.join(src, f), train_dest)

    for f in test_files:
        shutil.copy(os.path.join(src, f), test_dest)

split_and_copy(SOURCE_AI, TRAIN_AI, TEST_AI)
split_and_copy(SOURCE_REAL, TRAIN_REAL, TEST_REAL)

print(" Dataset split completed")

# =====================================
# IMAGE PARAMETERS
# =====================================
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 10

# =====================================
# DATA GENERATORS
# =====================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "final_dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    "final_dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =====================================
# LOAD MOBILENETV2 MODEL
# =====================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Important for CPU training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# TRAIN MODEL
# =====================================
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# =====================================
# SAVE MODEL
# =====================================
MODEL_PATH = "ai_vs_real_model.h5"
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")

# =====================================
# EVALUATION
# =====================================
test_generator.reset()
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)

true_labels = test_generator.classes

accuracy = accuracy_score(true_labels, predicted_labels)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["AI", "REAL"]))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))
