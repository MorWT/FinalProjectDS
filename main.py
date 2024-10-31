# main.py

from ultralytics import YOLO
import cv2
import numpy as np
import os

# Step 1: Load Dataset Path and Set Configurations
# Update the path to your dataset and YAML config file
DATASET_PATH = '/Users/mortzabari/Documents/GitHub/FinalProjectDS/Data'
CONFIG_PATH = os.path.join(DATASET_PATH, 'data.yaml')  # YAML config file path


# Step 2: Initialize and Train the YOLO Model
def train_model():
    """Train the YOLO model on a custom dataset."""
    model = YOLO('yolo11n.pt')  # Start with a pretrained model or replace with your model checkpoint
    model.train(data=CONFIG_PATH, epochs=100, imgsz=640, batch=16)
    model.save("best_model.pt")
    print("Model training complete and saved as 'best_model.pt'")


# Step 3: Load Trained Model for Prediction
def load_trained_model(model_path='best_model.pt'):
    """Load a trained YOLO model."""
    return YOLO(model_path)


# Nutritional information dictionary for food items (sample values)
nutritional_info = {
    'apple_pie': {'calories': 320, 'fat': 14, 'sugar': 20, 'protein': 2},
    'baby_back_ribs': {'calories': 500, 'fat': 30, 'sugar': 5, 'protein': 40},
    # Add more items as needed
}


# Step 4: Perform Object Detection on Images
def detect_food_in_image(image, model):
    results = model.predict(source=image)
    for result in results:
        for box in result.boxes:
            label = int(box.cls)
            food_item = model.names[label]
            if food_item in nutritional_info:
                nutrition = nutritional_info[food_item]
                print(f"Food: {food_item}")
                print(f"Calories: {nutrition['calories']} kcal")
                print(f"Fat: {nutrition['fat']} g")
                print(f"Sugar: {nutrition['sugar']} g")
                print(f"Protein: {nutrition['protein']} g")
                print("------")

                # Draw bounding box and nutrition on image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                nutrition_text = (f"{food_item}\nCalories: {nutrition['calories']} kcal\n"
                                  f"Fat: {nutrition['fat']} g, Sugar: {nutrition['sugar']} g, "
                                  f"Protein: {nutrition['protein']} g")
                cv2.putText(image, nutrition_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image


# Step 5: Perform Object Detection on Videos
def detect_food_in_video(video_path, model, save_processed=False):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Video at {video_path} not found or can't be opened.")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    if save_processed:
        output_path = 'processed_video.avi'
        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        processed_frame = detect_food_in_image(frame, model)

        cv2.imshow('Detected Food', processed_frame)
        if save_processed:
            output_video.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    if save_processed:
        output_video.release()
    cv2.destroyAllWindows()


# Main Execution
if __name__ == "__main__":
    # Check if a model has been trained, if not, train the model
    if not os.path.exists("best_model.pt"):
        print("Training model as 'best_model.pt' not found.")
        train_model()
    else:
        print("'best_model.pt' found. Skipping training.")

    # Load the trained model
    model = load_trained_model("best_model.pt")

    # Example usage for image detection
    image_path = os.path.join(DATASET_PATH, 'images', 'apple_pie', 'img1.jpg')  # Adjust as needed
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    processed_image = detect_food_in_image(image, model)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Example usage for video detection (press 'q' to quit video)
    video_path = os.path.join(DATASET_PATH, 'your_video.mp4')  # Update with your video file
    detect_food_in_video(video_path, model, save_processed=True)
