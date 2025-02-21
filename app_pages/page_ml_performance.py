import streamlit as st
import matplotlib.pyplot as plt
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