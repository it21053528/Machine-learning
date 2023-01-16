import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

 #image loading function
def load_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img

# Define the neural style transfer function
def neural_style_transfer(content_image, style_image):
    # Implement the neural style transfer algorithm here
   
    content_load = load_image(content_image)
    style_load = load_image(style_image)
   
    styled_output = model(tf.constant(content_load), tf.constant(style_load))[0]

    return styled_output

st.title("Neural Style Transfer App")

# Allow user to upload content and style images
content_Image = st.file_uploader("Choose a content image", type=["jpg", "png", "jpeg"])
if content_Image:
   st.image(content_Image, width=150)

style_Image = st.file_uploader("Choose a style image", type=["jpg", "png", "jpeg"])
if style_Image:
    st.image(style_Image, width=150)

# content_Image = 'profile Pic.jpg'
# style_Image = 'paper.jpg'

if content_Image and style_Image:
    # Apply neural style transfer
    styled_output = neural_style_transfer(content_Image, style_Image)

    # # Convert TensorFlow tensor to PIL Image
    stylized_image = tf.keras.preprocessing.image.array_to_img(styled_output)

    # Display the styled output
    st.image(styled_output, caption="Styled Output")
else:
    st.info("Please upload both content and style images")
