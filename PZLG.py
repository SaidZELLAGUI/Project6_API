# import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import keras
from PIL import Image
import cv2
import numpy as np

# Displaying in-head
st.title("DOG BREEDS")
st.header("Dogs breeds prediction. Open Classroom Projet6")
st.image("banner2.jpg")
st.text("Please upload the image that you want to know the breed in jpg format")

# Importing labels name
my_content = open("dogs_name.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()

def image_classifier(img, weights_file):
  """Function which classifies an image. 

  Inputs: Image to classify & Model to load

  Output : Prediction with the greater probability
  """
  model = keras.models.load_model(weights_file)
  image = img_to_array(img)
  image = cv2.resize(image,(224,224))
  image = image.reshape(1,224,224,3)
  predictions = model.predict(image)
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions)
  return dogs_list[predictions]

# Displaying uploader
uploaded_file = st.file_uploader("Upload your image...", type="jpg")

# Loop ending with prediction
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption='Uploaded image.', use_column_width=True)
  st.write("")
  st.write("Classifying...")
  label = image_classifier(img, 'My_model.h5')
  st.write(label)


st.markdown("*Created by Said ZELLAGUI, Machine Learning Engineer*")
