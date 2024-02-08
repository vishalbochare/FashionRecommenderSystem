import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

if 'target_language_choice' not in st.session_state:
    st.session_state['target_language_choice'] = 'en'

# Selectable for choosing target language
target_language_choice = st.selectbox('Choose target language', ['En', 'Es', 'Fr', 'Hi', 'Mr'])
st.session_state['target_language_choice'] = target_language_choice


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


@st.cache_data
def translate_text(text, target_language_choice='en'):
    if target_language_choice == 'en':
        return text

    import requests
    import uuid

    resource_key = '64461d5372e74cfbaf28f9d44f38eef7'

    region = 'eastus'

    endpoint = 'https://api.cognitive.microsofttranslator.com/'

    path = '/translate?api-version=3.0'
    params = f'&from=en&to={target_language_choice}'
    constructed_url = endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': resource_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': text
    }]

    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    # print(response[0]['translations'][0]['text'])
    if request.status_code==200:
        translated_text = response[0]['translations'][0]['text']
        return translated_text
    else:
        print(f"Error: {request.status_code}")
        return "ot"+text



st.title(translate_text('Fashion Recommender System', st.session_state['target_language_choice']))

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices



# steps
# file upload -> save
uploaded_file = st.file_uploader(translate_text('Choose an image', st.session_state['target_language_choice']))
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        #st.text(features)
        # recommendation
        indices = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.header(translate_text("One", st.session_state['target_language_choice']))
            st.image(filenames[indices[0][0]])
        with col2:
            st.header(translate_text("Two", st.session_state['target_language_choice']))
            st.image(filenames[indices[0][1]])
        with col3:
            st.header(translate_text("Three", st.session_state['target_language_choice']))
            st.image(filenames[indices[0][2]])
        with col4:
            st.header(translate_text("Four", st.session_state['target_language_choice']))
            st.image(filenames[indices[0][3]])
        with col5:
            st.header(translate_text("Five", st.session_state['target_language_choice']))
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred in file upload")