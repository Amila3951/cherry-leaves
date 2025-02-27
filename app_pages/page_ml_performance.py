import streamlit as st
import matplotlib.pyplot as plt
<<<<<<< HEAD
import pandas as pd
from src.machine_learning.evaluate_clf import load_test_evaluation

def run():
    """
    This function displays the ML performance page.
    """
    st.title("ML Performance Metrics")

    version = 'v2'  # Replace with your actual version number

    st.write("### Average Image Size in Dataset")
    try:
        average_image_size = plt.imread(f"outputs/{version}/avg_img_size.png")
        st.image(average_image_size, caption='Average Image Size')
        st.warning(
            "The average image size in the provided dataset is: \n\n"
            "* Width average: 256px \n"
            "* Height average: 256px"
        )
    except FileNotFoundError:
        st.warning("Average image size plot not found.")
    st.write("---")

    st.write("### Train, Validation, and Test Set: Labels Frequencies")
    try:
        labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
        st.image(labels_distribution, caption='Labels Distribution')
        st.success(
            f"* Train - healthy: 1472 images\n"
            f"* Train - powdery_mildew: 1472 images\n"
            f"* Validation - healthy: 210 images\n"
            f"* Validation - powdery_mildew: 210 images\n"
            f"* Test - healthy: 422 images\n"
            f"* Test - powdery_mildew: 422 images\n"
        )
    except FileNotFoundError:
        st.warning("Labels distribution plot not found.")
    st.write("---")

    st.write("### Model History")
    st.info(
        "The model learning curve is used to check the model for "
        "overfitting and underfitting by plotting loss and accuracy."
    )
    col1, col2 = st.columns(2)
    try:
        with col1:
            model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
            st.image(model_acc, caption='Model Training Accuracy')
        with col2:
            model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
            st.image(model_loss, caption='Model Training Losses')
    except FileNotFoundError:
        st.warning("Model history plots not found.")
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    evaluation = load_test_evaluation(version)
    if evaluation is None:
        st.warning("No model evaluation results available.")
    else:
        st.dataframe(pd.DataFrame(evaluation, index=['Loss', 'Accuracy']))
        st.write(f"> **The accuracy of the ML model is {evaluation['test_accuracy']*100:.0f}%**")
=======
from src.machine_learning.evaluate_clf import load_test_evaluation 

def run():
    st.title("Machine Learning Model Performance")

    st.write("### 📈 Model Evaluation")
    st.info(
        "**Objective:**\n"
        "This page presents the performance metrics of the trained machine learning model.\n"
        "These metrics are crucial for assessing the model's effectiveness in distinguishing"
        " between healthy and mildew-infected cherry leaves."
    )

    version = 'v1'

    # Load metrics
    evaluation = load_test_evaluation(version)

    # Display metrics
    if evaluation is None:
        st.warning(f"No model evaluation results available for version {version}. Please train the model and generate the evaluation report.")
    else:
        st.write(f"**Model Version:** {version}")
        st.write(f"- Accuracy: {evaluation['accuracy']:.2f}")
        st.write(f"- Precision: {evaluation['precision']:.2f}")
        st.write(f"- Recall: {evaluation['recall']:.2f}")
        st.write(f"- F1 Score: {evaluation['f1']:.2f}")

        # Display confusion matrix
        st.write("### 📊 Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(evaluation['confusion_matrix'], cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Healthy', 'Mildew'])
        ax.set_yticklabels(['Healthy', 'Mildew'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        # Display classification report
        st.write("### 📝 Classification Report")
        st.text(evaluation['classification_report'])
>>>>>>> 677e0a13154007f722449a0e0a1e43eb067945f9
