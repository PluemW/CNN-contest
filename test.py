import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Define paths
test_images_dir = 'Test_contest/Test Images' 
csv_file = 'Test_contest/test.csv'
output_csv = 'Test_contest/predictions.csv'  # Output path as per your original request

# Define the single food class and model file
model_file = 'best_model.keras'  # Model file for the chosen food class

# Define image size to match your model requirements
IMAGE_SIZE = (224, 224)

print(f"Using image size: {IMAGE_SIZE}")
print(f"Checking paths...")
print(f"CSV file exists: {os.path.exists(csv_file)}")
print(f"Test images directory exists: {os.path.exists(test_images_dir)}")

# Load the model with error handling
def load_model_safely(model_path):
    try:
        print(f"Loading model: {model_path}")
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

# Load the single model for the chosen food class
model_path = os.path.join(model_file)
model = load_model_safely(model_path)
if model is None:
    print("Failed to load the model. Exiting...")
    exit()

# Function to list available test images
def list_test_images():
    return os.listdir(test_images_dir) if os.path.exists(test_images_dir) else []

available_images = list_test_images()
print(f"Number of available test images: {len(available_images)}")

# Function to preprocess image
def preprocess_image(image_path):
    full_path = os.path.join(test_images_dir, image_path) if not os.path.exists(image_path) else image_path
    
    if not os.path.exists(full_path):
        print(f"Image not found: {image_path}")
        return None
    
    try:
        img = load_img(full_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to get food score using the single model
def get_food_score(image_path):
    if model is None:
        return 0
    
    img_array = preprocess_image(image_path)
    if img_array is None:
        return 0
    
    try:
        prediction = model.predict(img_array)
        return float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
    except Exception as e:
        print(f"Error getting score for {image_path}: {e}")
        return 0

# Parse and process CSV data
def parse_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return pd.DataFrame()

def process_comparison_data():
    df = parse_csv_file(csv_file)
    if df.empty:
        return df
    
    results = []
    for _, row in df.iterrows():
        image1, image2, actual_winner = str(row['Image 1']), str(row['Image 2']), int(row['Winner'])
        
        # Get scores for both images using the single model
        score1 = get_food_score(image1)
        score2 = get_food_score(image2)
        
        # Predict the winner based on scores
        predicted_winner = 1 if score1 > score2 else 2
        
        results.append({
            'Image1': image1, 'Score1': score1,
            'Image2': image2, 'Score2': score2,
            'PredictedWinner': predicted_winner, 'ActualWinner': actual_winner,
            'Correct': predicted_winner == actual_winner
        })
    
    return pd.DataFrame(results)

def run_comparison_system():
    results_df = process_comparison_data()
    if results_df.empty:
        return results_df
    
    accuracy = results_df['Correct'].mean() * 100
    print(f"Overall accuracy: {accuracy:.2f}%")
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return results_df

def visualize_examples(results_df, num_examples=3):
    if results_df.empty:
        return
    
    examples = results_df.sample(min(num_examples, len(results_df)))
    
    for _, example in examples.iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        for i, (img_col, score_col) in enumerate([
            ('Image1', 'Score1'), ('Image2', 'Score2')
        ]):
            img_path = os.path.join(test_images_dir, example[img_col])
            
            if os.path.exists(img_path):
                img = load_img(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Score: {example[score_col]:.2f}")
            else:
                axes[i].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i].axis('off')
        
        plt.suptitle(f"Predicted: {example['PredictedWinner']}, Actual: {example['ActualWinner']} {'✓' if example['Correct'] else '✗'}")
        plt.show()

if __name__ == "__main__":
    results = run_comparison_system()
    if not results.empty:
        visualize_examples(results)