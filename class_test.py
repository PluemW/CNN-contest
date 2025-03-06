import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define constants
MODEL_PATH = "food_classifier_mobilenet.keras"
TEST_IMAGE_DIR = "test_images/Test Images"  # Change this path if needed
IMAGE_SIZE = (224, 224)

# Define class mapping
CLASS_MAPPING = {
    0: 'burger',
    1: 'dessert',
    2: 'pizza',
    3: 'ramen',
    4: 'sushi'
}

# Load trained model
model = keras.models.load_model(MODEL_PATH)

def predict_and_show_image(image_path):
    """ Predict the class of an image and display the result. """
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100

    # Display image with prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {CLASS_MAPPING[predicted_class]} ({confidence:.2f}%)", fontsize=14, fontweight='bold')
    plt.show()

def test_all_images(directory):
    """ Loop through all images in a folder and classify them. """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in the directory.")
        return

    print(f"Found {len(image_files)} images. Classifying...\n")

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        print(f"Processing: {image_file}")
        predict_and_show_image(image_path)

# Run the test on all images in the folder
if __name__ == "__main__":
    test_all_images(TEST_IMAGE_DIR)
