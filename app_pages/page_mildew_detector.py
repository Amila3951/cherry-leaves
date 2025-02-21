import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def run():
    st.title("Cherry Leaf Mildew Detector")

    st.write("### 🔍 Analyze Cherry Leaves")
    st.info(
        "**Objective:**\n"
        "Upload cherry leaf images to predict if they are healthy or infected with powdery mildew.\n\n"
    )

    st.write("### 📥 Upload Images")
    st.markdown(
        "- Download sample images from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).\n"
        "- Upload one or more **PNG** images below."
    )

    images_buffer = st.file_uploader(
        label="📤 Upload cherry leaf images (PNG format only)",
        type='png',
        accept_multiple_files=True
    )

    if images_buffer:
        df_report = pd.DataFrame()
        for image in images_buffer:
            st.subheader(f"📄 Leaf Sample: **{image.name}**")
            img_pil = Image.open(image)
            st.image(img_pil, caption=f"🖼️ Image: {image.name}")

            version = 'v1'  
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)

            st.success(f"**Prediction:** {pred_class.upper()} ({pred_proba * 100:.2f}%)")
            plot_predictions_probabilities(pred_proba, pred_class)

            new_row = pd.DataFrame([{
                "Image Name": image.name,
                "Prediction": pred_class,
                "Confidence": f"{pred_proba * 100:.2f}%"
            }])
            df_report = pd.concat([df_report, new_row], ignore_index=True)

        st.write("---")
        if not df_report.empty:
            st.success("### 📊 Prediction Summary Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
            st.info("Download the table as a CSV file.")