import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

features_array = np.vstack(features_list)

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

st.title('Clothing recommender system')


def save_file(uploaded_file):
    try:
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normalized = flatten_result / norm(flatten_result)

    return result_normalized


def recommend(result_normalized, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_array)

    distance, indices = neighbors.kneighbors([result_normalized])

    return indices

uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_file(uploaded_file):
        # display image
        uploaded_img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        uploaded_img = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)

        # extract features of uploaded image
        result_normalized = extract_img_features(uploaded_file.name, model)
        st.text(result_normalized)
        img_indices = recommend(result_normalized, features_array)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.header("I")
            st.image(img_files_list[img_indices[0][0]])

        with col2:
            st.header("II")
            st.image(img_files_list[img_indices[0][1]])

        with col3:
            st.header("III")
            st.image(img_files_list[img_indices[0][2]])

        with col4:
            st.header("IV")
            st.image(img_files_list[img_indices[0][3]])

        with col5:
            st.header("V")
            st.image(img_files_list[img_indices[0][4]])
    else:
        st.header("Some error occurred")
