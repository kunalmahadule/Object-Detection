# Original Code

import os
import cv2
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from ultralytics import YOLO  # Using YOLOv8 for advanced detection

# Paths
IMAGES_FOLDER = "images"
RESULT_FOLDER = "result"
MODEL_PATH = "yolov8x.pt"  # Using YOLOv8x for high-accuracy detection
CSV_PATH = os.path.join(RESULT_FOLDER, "detected_objects.csv")

# Create result folder if it doesn't exist
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model with high confidence threshold
model = YOLO(MODEL_PATH)

# Function to generate random dates
def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

# Generate manufacturing and expiry dates
def generate_dates():
    manufacturing_date = random_date(datetime(2024, 6, 1), datetime(2025, 2, 1)).strftime("%Y-%m-%d")
    expiry_date = random_date(datetime(2025, 1, 1), datetime(2026, 12, 31)).strftime("%Y-%m-%d")
    return manufacturing_date, expiry_date

# Object name mapping for more attractive names
object_name_mapping = {
    "bottle": ["Coca-Cola", "Pepsi", "Sprite", "Fanta", "Wine", "Sauce"],
    "fridge": ["Samsung Refrigerator", "LG Fridge", "Whirlpool Cooler", "Godrej Freezer"],
    "laptop": ["Dell Inspiron", "HP Pavilion", "MacBook Pro", "Lenovo ThinkPad"],
    "phone": ["iPhone 14", "Samsung Galaxy S23", "OnePlus 11", "Google Pixel 7"]
}

# Random descriptions
random_descriptions = [
    "High-quality product with premium materials.",
    "Limited edition item with exclusive features.",
    "Best-selling product with high durability.",
    "Eco-friendly and sustainable product design.",
    "Latest model with advanced technology.",
    "Budget-friendly with great performance."
]

# Data storage
data = []

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

    # Perform object detection with higher confidence threshold
    results = model(image, conf=0.5)  # Increase confidence threshold to improve accuracy

    # Draw bounding boxes and labels on the image
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box[:6].tolist()
            label = model.names[int(class_id)]

            # Filter out low-confidence detections
            if confidence < 0.5:
                continue

            # Ignore detected persons from CSV
            if label.lower() == "person":
                continue

            # Replace generic object names with more attractive names
            if label in object_name_mapping:
                label = random.choice(object_name_mapping[label])

            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw label with better visibility
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(image, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Generate random manufacturing and expiry dates
            mfg_date, exp_date = generate_dates()

            # Generate random product description
            description = random.choice(random_descriptions)

            # Append data for CSV
            data.append([label, mfg_date, exp_date, description])

    # Save the result
    result_path = os.path.join(RESULT_FOLDER, image_name)
    cv2.imwrite(result_path, image)
    print(f"Processed and saved: {result_path}")

# Save detected objects to CSV
columns = ["Product Name", "Manufacturing Date", "Expiry Date", "Description"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(CSV_PATH, index=False)
print(f"Object detection completed with improved accuracy. Data saved to {CSV_PATH}.")