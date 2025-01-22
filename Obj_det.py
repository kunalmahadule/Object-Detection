'''
import os
import cv2
import numpy as np
from ultralytics import YOLO  # Using YOLOv8 for simplicity

# Paths
IMAGES_FOLDER = "images"
RESULT_FOLDER = "result"
MODEL_PATH = "yolov8n.pt"  # Pre-trained YOLOv8 model

# Create result folder if it doesn't exist
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Process each image in the images folder
for image_name in os.listdir(IMAGES_FOLDER):
    image_path = os.path.join(IMAGES_FOLDER, image_name)
    
    # Check if it's a valid image
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_name}")
        continue

    # Perform object detection
    results = model(image)

    # Draw bounding boxes and labels on the image
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box[:6].tolist()
            label = model.names[int(class_id)]

            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw label
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(image, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result
    result_path = os.path.join(RESULT_FOLDER, image_name)
    cv2.imwrite(result_path, image)
    print(f"Processed and saved: {result_path}")

print("Object detection completed.")
'''

'''Improved Code'''

import os
import cv2
import numpy as np
from ultralytics import YOLO  # Using YOLOv8 for simplicity

# Paths
IMAGES_FOLDER = "images"
RESULT_FOLDER = "result"
MODEL_PATH = "yolov8x.pt"  # Using YOLOv8x for more comprehensive detection

# Create result folder if it doesn't exist
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Process each image in the images folder
for image_name in os.listdir(IMAGES_FOLDER):
    image_path = os.path.join(IMAGES_FOLDER, image_name)
    
    # Check if it's a valid image
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_name}")
        continue

    # Perform object detection
    results = model(image)

    # Draw bounding boxes and labels on the image
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box[:6].tolist()
            label = model.names[int(class_id)]

            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw label
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(image, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result
    result_path = os.path.join(RESULT_FOLDER, image_name)
    cv2.imwrite(result_path, image)
    print(f"Processed and saved: {result_path}")

print("Object detection completed.")
