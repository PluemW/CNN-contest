import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Enable mixed precision for performance boost on GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define constants
DATA_DIR = "class"  # Folder with subdirectories (Burger, Dessert, etc.)
IMAGE_SIZE = (224, 224)  # MobileNetV2 input size
BATCH_SIZE = 16
EPOCHS = 20

# Ensure GPU is being used
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Data augmentation & preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

# Load training data
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",  # Auto-labels classes
    subset="training"
)

# Load validation data
val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

# Load MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Freeze the base model initially
base_model.trainable = False

# Build model with MobileNetV2 base
model = Sequential([
    base_model,  # Add the pre-trained MobileNetV2 base model
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')  # Dynamic class count
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save model
model.save("food_classifier_mobilenet.keras")

print("Training complete. Model saved as 'food_classifier_mobilenet.keras'")
