import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from PIL import ImageOps

import cv2

def enhanced_preprocess_image(image_path):
    original_img = Image.open(image_path)

    # Convert images with transparency to an opaque format
    if original_img.mode in ('RGBA', 'LA', 'P'):
        background_layer = Image.new("RGB", original_img.size, "WHITE")
        background_layer.paste(original_img, mask=original_img.getchannel('A') if original_img.mode == 'RGBA' else original_img)
        processed_img = background_layer
    else:
        processed_img = original_img.convert('RGB')

    # Convert to grayscale and apply inversion for digit clarity
    processed_img = ImageOps.grayscale(processed_img)
    processed_img = ImageOps.invert(processed_img)

    # Calculate the new size keeping aspect ratio constant
    aspect = min(28 / processed_img.width, 28 / processed_img.height)
    new_dimensions = (int(processed_img.width * aspect), int(processed_img.height * aspect))
    processed_img = processed_img.resize(new_dimensions, Image.Resampling.LANCZOS)

    # Place the processed image onto a centered canvas
    canvas = Image.new('L', (28, 28), 'white')
    paste_coords = ((28 - new_dimensions[0]) // 2, (28 - new_dimensions[1]) // 2)
    canvas.paste(processed_img, paste_coords)

    # Prepare the image array for the model
    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_array = img_array.reshape(-1, 28, 28, 1)  # Ensuring proper dimensions for the model

    return img_array
# Function to predict the digit from an image

def predict_digit(model, image):
    prediction = model.predict(image)
    print("prediction:", prediction)
    return np.argmax(prediction), np.max(prediction)

# Load the pre-trained model (make sure the path to the model is correct)
def load_pre_train():
    model_path = 'digit_classifier_model.h5'
    model = load_model(model_path)
    return model

# from tensorflow.keras.datasets import mnist
# import numpy as np
# from model import load_model, make_prediction

# # Load MNIST data
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# # Preprocess the data similarly to how the model was trained
# test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# # Load your trained model - replace 'path_to_your_model.h5' with the actual path
# model = load_model('digital_classifier_model.h5')

# # Predict the first 10 images from the test set
# for i in range(10):
#     image = np.expand_dims(test_images[i], axis=0)  # Expanding dims to match the input shape
#     predicted_class, confidence = make_prediction(model, image)
#     print(f"True Label: {test_labels[i]}, Predicted Label: {predicted_class}, Confidence: {confidence}")