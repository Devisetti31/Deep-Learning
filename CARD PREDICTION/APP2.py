import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

result = {'ace of clubs': 0, 'ace of diamonds': 1, 'ace of hearts': 2, 'ace of spades': 3, 'eight of clubs': 4, 'eight of diamonds': 5,
          'eight of hearts': 6, 'eight of spades': 7, 'five of clubs': 8, 'five of diamonds': 9, 'five of hearts': 10, 'five of spades': 11,
          'four of clubs': 12, 'four of diamonds': 13, 'four of hearts': 14, 'four of spades': 15, 'jack of clubs': 16, 'jack of diamonds': 17,
          'jack of hearts': 18, 'jack of spades': 19, 'joker': 20, 'king of clubs': 21, 'king of diamonds': 22, 'king of hearts': 23,
          'king of spades': 24, 'nine of clubs': 25, 'nine of diamonds': 26, 'nine of hearts': 27, 'nine of spades': 28, 'queen of clubs': 29,
          'queen of diamonds': 30, 'queen of hearts': 31, 'queen of spades': 32, 'seven of clubs': 33, 'seven of diamonds': 34, 'seven of hearts': 35,
          'seven of spades': 36, 'six of clubs': 37, 'six of diamonds': 38, 'six of hearts': 39, 'six of spades': 40, 'ten of clubs': 41,
          'ten of diamonds': 42, 'ten of hearts': 43, 'ten of spades': 44, 'three of clubs': 45, 'three of diamonds': 46, 'three of hearts': 47,
          'three of spades': 48, 'two of clubs': 49, 'two of diamonds': 50, 'two of hearts': 51, 'two of spades': 52}

st.title("Cards Prediction")

# Load the model
final_model = load_model("cards_vgg16.h5")

# Upload image
image = st.file_uploader("Upload input Image", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

if image is not None:
    # Display the uploaded image
    st.write('### Your Selected Image Is : ')
    st.image(image, caption='Your Image', use_column_width=False)

    # Convert the uploaded image to a format suitable for prediction
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # Load the image using OpenCV

    if img is None:
        st.write("Error: Unable to load image")
    else:
        # Preprocess the image
        img_resized = cv2.resize(img, (224, 224))  # Resize the image
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_array = img_to_array(img_rgb)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
        img_array = img_array / 255.0  # Normalize the image

        if st.button('PREDICT'):
            # Make prediction
            prediction = final_model.predict(img_array)
            prediction = np.argmax(final_model.predict(cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)[np.newaxis]))
            #st.write(f"Prediction: {prediction}")
            value = {i for i in result if result[i]==prediction}
            prediction_text = f'<span style="color:blue; font-size:27px;">{value}</span>'

            st.markdown(f"### Your Predicted Image is : {prediction_text} 😁", unsafe_allow_html=True)
