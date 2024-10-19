import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from lime import lime_image
import numpy as np
import plotly.graph_objects as go
from utils.loss_and_scoring import FocalLoss, f1_score

IMG_SIZE = (300, 300)
THRESHOLD = 0.53

def preprocess_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize_with_pad(image, target_height=IMG_SIZE[0], target_width=IMG_SIZE[1])
    image = tf.expand_dims(image, axis=0)

    return image


def explain_image(image, model):
    explainer = lime_image.LimeImageExplainer()
    image_resized = tf.image.resize_with_pad(image, target_height=IMG_SIZE[0], target_width=IMG_SIZE[1])
    
    explanation = explainer.explain_instance(image_resized.numpy(), lambda x: model.predict(x, verbose=0), 
                                             top_labels=5, hide_color=0, num_samples=1500, random_seed=50)

    label_to_explain = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(label_to_explain, positive_only=True, num_features=10, hide_rest=False)

    temp = (temp - temp.min()) / (temp.max() - temp.min())
    temp = (temp * 255).astype(np.uint8)

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_resized.numpy()))
    fig.add_trace(go.Heatmap(z=mask, colorscale='Jet', opacity=0.5, showscale=False))

    fig.update_layout(title=f'LIME Explanation for Label {label_to_explain}', 
                      xaxis=dict(showgrid=False, showticklabels=False, ticks=""),
                      yaxis=dict(showgrid=False, showticklabels=False, ticks=""))

    return fig


@st.cache_resource
def load_model():
    return keras.models.load_model('cnn_fraud_model_effv2b3.keras', custom_objects={'FocalLoss': FocalLoss, 'f1_score': f1_score})


if 'model' not in st.session_state:
    st.session_state.model = load_model()

model = st.session_state.model

if model is None:
    st.stop()

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

st.title("Vehicle Insurance Fraud Model")

with st.form("my-form", clear_on_submit=False):
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Classify Image")

if submitted and uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_resized = tf.image.resize_with_pad(image, target_height=IMG_SIZE[0], target_width=IMG_SIZE[1])
    preprocessed_image = preprocess_image(image)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Image(z=image_resized.numpy()))
        fig.update_layout(title='Uploaded Image',
                          xaxis=dict(showgrid=False, showticklabels=False, ticks=""),
                          yaxis=dict(showgrid=False, showticklabels=False, ticks=""))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with st.spinner('Classifying image...'):
            fraud_probability = model.predict(preprocessed_image)[0][1]

        if fraud_probability >= THRESHOLD:
            st.write(f"Fraud detected with {fraud_probability:.2f} probability!")
        else:
            st.write(f"No fraud detected. Fraud probability: {fraud_probability:.2f}")

    with col2:
        with st.spinner('Generating LIME Plot...'):
            lime_fig = explain_image(image, model)
            st.plotly_chart(lime_fig, use_container_width=True, config={'displayModeBar': False})

    st.session_state.uploaded_file = None
