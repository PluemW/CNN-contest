import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import MobileNetV2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

# Enable mixed precision for performance boost
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Constants
BATCH_SIZE = 24
IMAGE_SIZE = (224, 224)
EPOCHS = 50
DATA_DIR = "train/Questionair_Images"
CSV_FILE = "train/data_from_questionaire.csv"
MODEL_SAVE_DIR = "saved_models_each_3"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Ensure GPU is being used
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load dataset
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing CSV file: {CSV_FILE}")
dataframe = pd.read_csv(CSV_FILE)

# Function to check if image paths exist
def verify_image_paths(row):
    img1_path = os.path.join(DATA_DIR, row['Image 1'])
    img2_path = os.path.join(DATA_DIR, row['Image 2'])
    return os.path.exists(img1_path) and os.path.exists(img2_path)

# Filter missing images
dataframe = dataframe[dataframe.apply(verify_image_paths, axis=1)].reset_index(drop=True)

# Extract food type from filename
def get_food_type(filename):
    prefix = filename[0].lower()
    food_types = {'s': 'Sushi', 'r': 'Ramen', 'b': 'Burger', 'p': 'Pizza', 'd': 'Dessert'}
    return food_types.get(prefix, 'Unknown')

dataframe['Food Type'] = dataframe['Image 1'].apply(get_food_type)

# Warn about unknown food types
unknown_types = dataframe[dataframe['Food Type'] == 'Unknown']
if not unknown_types.empty:
    print(f"Warning: {len(unknown_types)} images have unknown food types. Check dataset formatting.")

def prepare_data_for_food_type(df, food_type):
    food_df = df[df['Food Type'] == food_type].reset_index(drop=True)
    images, labels = [], []
    
    for _, row in food_df.iterrows():
        img1_path = os.path.join(DATA_DIR, row['Image 1'])
        img2_path = os.path.join(DATA_DIR, row['Image 2'])
        
        img1 = img_to_array(load_img(img1_path, target_size=IMAGE_SIZE)) / 255.0
        img2 = img_to_array(load_img(img2_path, target_size=IMAGE_SIZE)) / 255.0
        
        images.append(img1)
        images.append(img2)
        
        labels.append(1 if row['Winner'] == 1 else 0)
        labels.append(0 if row['Winner'] == 1 else 1)
    
    return np.array(images), np.array(labels)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)

# Learning rate schedule
def lr_schedule(epoch):
    return 1e-3 if epoch < 10 else 5e-4 if epoch < 20 else 1e-5

lr_callback = LearningRateScheduler(lr_schedule)

# Focal Loss for imbalanced data
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * K.pow((1 - p_t), gamma) * bce
    return loss

# Model creation function
def create_food_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Enable training on base model
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

def train_model_for_food_type(food_type):
    print(f"\n============ Training model for {food_type} ============")
    
    X, y = prepare_data_for_food_type(dataframe, food_type)
    
    if len(X) == 0:
        print(f"No data found for {food_type}. Skipping...")
        return None
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    
    model = create_food_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss(), metrics=['accuracy'])
    
    model_path = os.path.join(MODEL_SAVE_DIR, f"{food_type.lower()}_model.keras")
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=f'./logs/{food_type}')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    history = model.fit(
        train_generator, validation_data=(X_val, y_val),
        epochs=EPOCHS, callbacks=[checkpoint, tensorboard, lr_scheduler, lr_callback, early_stopping]
    )
    
    print(f"Model for {food_type} saved to {model_path}")
    return model

food_types = ['Sushi', 'Ramen', 'Burger', 'Pizza', 'Dessert']
trained_models = {food_type: train_model_for_food_type(food_type) for food_type in food_types if train_model_for_food_type(food_type)}