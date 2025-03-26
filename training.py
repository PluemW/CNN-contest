import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Input, BatchNormalization,
                                    Conv2D, MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and ready for training.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, training will run on CPU.")

# Define constants
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
EPOCHS = 20
DATA_DIR = "train/Questionair_Images"
CSV_FILE = "train/data_from_questionaire.csv"
MODEL_SAVE_PATH = "best_model.keras"
os.makedirs("saved_models", exist_ok=True)

# Load dataset
import pandas as pd
dataframe = pd.read_csv(CSV_FILE)

def verify_image_paths(row):
    img1_path = os.path.join(DATA_DIR, row['Image 1'])
    img2_path = os.path.join(DATA_DIR, row['Image 2'])
    return os.path.exists(img1_path) and os.path.exists(img2_path)

dataframe = dataframe[dataframe.apply(verify_image_paths, axis=1)].reset_index(drop=True)

if dataframe.empty:
    raise ValueError("Error: No valid image pairs found after filtering. Check your CSV and image paths!")

# Load images
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0  # Normalize to [0,1]
    return img

# Prepare data
images = []
labels = []

for _, row in dataframe.iterrows():
    img1_path = os.path.join(DATA_DIR, row['Image 1'])
    img2_path = os.path.join(DATA_DIR, row['Image 2'])
    
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    
    images.append(img1)
    images.append(img2)
    
    if row['Winner'] == 1:
        labels.append(1)  # First image is winner
        labels.append(0)  # Second image is not winner
    else:
        labels.append(0)  # First image is not winner
        labels.append(1)  # Second image is winner

X = np.array(images)
y = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define CNN model using Conv2D
def create_model(input_shape=(224, 224, 3)):
    input_img = Input(shape=input_shape)

    # Feature extraction
    x = GlobalAveragePooling2D()(input_img)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=input_img, outputs=outputs)
    return model

# Compile and train the model
model = create_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
tensorboard = TensorBoard(log_dir='./logs/all_foods')

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, lr_scheduler, tensorboard]
)

print(f"Best model saved to {MODEL_SAVE_PATH}")
