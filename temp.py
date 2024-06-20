import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)


# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to predict components in the image using YOLOv5
def predict_components(image):
    img = np.array(image)  # Convert PIL image to numpy array
    results = model(img)  # Make predictions
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    img = np.array(image)
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img

# Streamlit application title and file uploader
st.title("Image Component Detection")
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    img = load_image(image_file)  # Load the image
    st.image(image_file, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image
    
    if st.button("Analyse Image"):  # Analyse button
        results = predict_components(img)  # Predict components
        img_with_boxes = draw_boxes(img, results)  # Draw bounding boxes
        st.image(img_with_boxes, caption='Detected Components', use_column_width=True)  # Display the image with boxes
        st.write("Detected Components:")  # Display the predictions
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            st.write(f"{label} - Confidence: {conf:.2f}")

