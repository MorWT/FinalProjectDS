import os
import shutil
import random

# Paths
DATASET_PATH = '/Users/mortzabari/Documents/GitHub/FinalProjectDS/Data/images'
OUTPUT_PATH = '/Users/mortzabari/Documents/GitHub/FinalProjectDS/Data/split_images'
LABELS_PATH = os.path.join('/Users/mortzabari/Documents/GitHub/FinalProjectDS/Data/meta', 'labels.txt')

# Create output directories
os.makedirs(os.path.join(OUTPUT_PATH, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'test'), exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Function to split dataset
def split_dataset(dataset_path):
    food_types = os.listdir(dataset_path)  # List of food type folders
    for food_type in food_types:
        food_type_path = os.path.join(dataset_path, food_type)
        if os.path.isdir(food_type_path):
            images = os.listdir(food_type_path)
            random.shuffle(images)  # Shuffle images for randomness

            # Calculate split indices
            train_size = int(len(images) * TRAIN_RATIO)
            val_size = int(len(images) * VAL_RATIO)

            # Split images
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]

            # Move images to respective directories
            for img in train_images:
                shutil.copy(os.path.join(food_type_path, img), os.path.join(OUTPUT_PATH, 'train', food_type))
            for img in val_images:
                shutil.copy(os.path.join(food_type_path, img), os.path.join(OUTPUT_PATH, 'val', food_type))
            for img in test_images:
                shutil.copy(os.path.join(food_type_path, img), os.path.join(OUTPUT_PATH, 'test', food_type))

# Execute the split
split_dataset(DATASET_PATH)
print("Dataset split into train, val, and test folders.")