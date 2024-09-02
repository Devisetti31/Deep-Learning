import numpy as np
import tensorflow
import keras
import cv2
from skimage import io
import streamlit as st

res = {0:'WithMask', 1:'WithoutMask'}

st.title('🖼️ FACE MASK PREDICTION USING VGG16 😷')

#final_model = tensorflow.keras.models.load_model("mask_detection_vgg16.h5")

from tensorflow.keras.models import load_model

final_model = load_model("mask_detection_vgg16.h5")


image = st.file_uploader("Upload input Image", accept_multiple_files=False)


#face_model = cv2.CascadeClassifier(r"C:\Users\LAKSHMI NARASIMHARAO\innomatics\Deep Learning\opencv\haarcascade_frontalface_alt.xml")

#for x,y,w,h in face_model.detectMultiScale(image):
if st.button('PREDICT'):
    st.write('### Your Selected Image Is : ')
    st.image(image, caption='Your Image', use_column_width=False)
    print(final_model.predict(cv2.resize(image,(224,224),interpolation=cv2.INTER_LINEAR)[np.newaxis]))
    if np.argmax(final_model.predict(cv2.resize(image,(224,224))[np.newaxis])) == 0:
        prediction = f'<span style="color:blue; font-size:27px;">{res[0].upper()}</span>'

        st.markdown(f"### Your Predicted Image is : {prediction} 😷", unsafe_allow_html=True)

    else:
        prediction = f'<span style="color:blue; font-size:27px;">{res[1].upper()}</span>'
        st.markdown(f"### Your Predicted Image is : {prediction} 😁", unsafe_allow_html=True)

