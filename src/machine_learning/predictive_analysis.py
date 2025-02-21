import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file

def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results.
    """

    class_labels = ['Healthy', 'Infected']
    probabilities = [1 - pred_proba, pred_proba]

    prob_per_class = pd.DataFrame({
        'Condition': class_labels,
        'Probability': probabilities
    })

    fig = px.bar(
        prob_per_class,
        x='Condition',
        y='Probability',
        range_y= [], 
        width=600,
        height=300,
        template='seaborn',
        color='Condition',
        title=f"Prediction Confidence: {pred_class.upper()}"
    )
    st.plotly_chart(fig)

def resize_input_image(img, version):
    """
    Reshape image to average image size.
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape, image_shape), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0) / 255.0
    return my_image

def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction.
    """
    model = load_model(f"outputs/{version}/cherry_leaves_model.h5")
    pred_proba = model.predict(my_image)
    pred_class = "Infected" if pred_proba > 0.5 else "Healthy"
    return pred_proba, pred_class