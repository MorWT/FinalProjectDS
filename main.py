import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm


# Load configuration parameters
def load_params(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


params = load_params('config/config_params.yaml')

# Paths from params.yaml
DATA_YAML_PATH = params['data']['data_yaml_path']  # Path to data.yaml
OUTPUT_SPLIT_DATA_PATH = params['data']['output_split_data_path']
OUTPUT_PATH = params['data']['output_path']

# Check if output directory exists, if not, create it
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Step 1: Initialize the YOLO Model
# Start with a pretrained model or replace with your model checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"======== Using device: {device} ========" )
model = YOLO('yolo11n.pt').to(device)


# Step 2: Load the Dataset
# YOLO model will automatically use paths in data.yaml for training and validation
def train_model():
    # Train the model using parameters in data.yaml
    model.train(data=DATA_YAML_PATH, epochs=10, imgsz=640, project=OUTPUT_PATH, name='train_results')


# Step 3: Perform Object Detection on Images
def detect_on_images(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for image_name in tqdm(os.listdir(image_folder), desc="Processing Images"):
        image_path = os.path.join(image_folder, image_name)
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            results = model.predict(source=image_path, save=True, project=output_folder)
            print(f"Detections saved for {image_name} in {output_folder}")


# # Step 4: Perform Object Detection on Videos
# def detect_on_video(video_path, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     video_name = Path(video_path).stem
#     output_video_path = os.path.join(output_folder, f"{video_name}_output.mp4")
#     results = model.predict(source=video_path, save=True, project=output_folder)
#     print(f"Processed video saved as {output_video_path}")


# Step 5: Answer Research Questions Through Experimentation
# This could involve logging model performance or interpreting YOLO’s confidence scores on specific classes.
def analyze_results():
    results = model.val()  # Validation metrics are computed automatically
    print(f"Validation results: {results}")

# Optional: Save Processed Video with Detections (using OpenCV for additional customization)
# def save_processed_video(input_video_path, output_video_path):
#     cap = cv2.VideoCapture(input_video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Perform object detection on each frame
#         results = model.predict(frame)
#         annotated_frame = np.squeeze(results.render())  # Get annotated frame
#         out.write(annotated_frame)
#
#     cap.release()
#     out.release()
#     print(f"Processed video saved at {output_video_path}")


# Run steps
if __name__ == "__main__":
    # Train the model
    print("Training the model...")
    train_model()

    # Image detection
    test_images_folder = os.path.join(OUTPUT_SPLIT_DATA_PATH, 'test')
    print("Detecting objects on test images...")
    detect_on_images(test_images_folder, os.path.join(OUTPUT_PATH, 'image_detections'))

    # # Video detection (if there are test videos)
    # test_video_path = '/path/to/test_video.mp4'  # Update with actual test video path
    # print("Detecting objects on test video...")
    # detect_on_video(test_video_path, os.path.join(OUTPUT_PATH, 'video_detections'))

    # Analyze results
    print("Analyzing results...")
    analyze_results()

    # # Optional: Save processed video with detections
    # processed_video_output_path = os.path.join(OUTPUT_PATH, 'processed_video.mp4')
    # save_processed_video(test_video_path, processed_video_output_path)

    print("All steps completed.")
