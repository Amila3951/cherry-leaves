import streamlit as st

def run():
    st.title("Project Hypothesis and Validation")

    st.write("### 📋 Project Hypothesis")
    st.info(
        "Our hypothesis is that cherry leaves infected with powdery mildew will exhibit "
        "distinguishable visual characteristics, such as discoloration or texture changes, "
        "which can be effectively detected through image analysis and machine learning."
    )

    st.write("### 🔎 Validation Approach")
    st.markdown(
        "To validate our hypothesis, we will follow these steps:\n\n"
        "1. **Visual Exploration:** Analyze images of healthy and infected leaves to identify potential visual cues.\n"
        "2. **Machine Learning:** Train a model to classify leaves as healthy or infected based on image data.\n"
        "3. **Evaluation:** Assess the model's performance using metrics such as accuracy, precision, and recall.\n"
        "4. **Analysis:** Investigate any misclassifications to understand limitations and potential areas for improvement."
    )

    st.write("### 📊 Current Findings")
    st.warning(
        "Preliminary findings suggest that while visual differences may be apparent in some cases, "
        "it can be challenging to reliably distinguish between healthy and infected leaves based on visual inspection alone. "
        "This highlights the need for a more robust approach, such as machine learning, to achieve accurate and consistent detection."
    )

    st.write("### ⏭️ Next Steps")
    st.markdown(
        "1. **Feature Engineering:** Explore additional image features and preprocessing techniques to enhance model performance.\n"
        "2. **Model Refinement:** Experiment with different model architectures and hyperparameters to optimize accuracy.\n"
        "3. **Further Validation:** Test the model on a larger and more diverse dataset to ensure generalizability."
    )