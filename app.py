import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Liver Disease Diagnostic System", layout="wide")

@st.cache_resource
def load_model():
    with open('liver_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("Liver Disease Prediction Model")
st.markdown("Early detection and supportive decision-making powered by Machine Learning.")
st.divider()

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("Dataset Insights")
    st.info("""
    This diagnostic model was trained on the Indian Liver Patient Dataset (ILPD). 
    The statistics below reflect the historical clinical data used to train the Random Forest algorithm.
    """)
    
    st.subheader("Training Data Overview")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric(label="Total Patients", value="583")
    metric_col2.metric(label="Liver Disease Cases", value="416", delta="71.3%", delta_color="off")
    
    metric_col3, metric_col4 = st.columns(2)
    metric_col3.metric(label="Avg. Patient Age", value="44.7")
    metric_col4.metric(label="Male to Female Ratio", value="3.1 : 1")
    
    st.divider()
    
    st.subheader("How It Works")
    st.markdown("""
    1. **Input Data:** Enter the patient's age, gender, and biochemical markers into the diagnostic form.
    2. **Process:** The system feeds the parameters into the trained Machine Learning algorithm.
    3. **Evaluate:** The model calculates the patterns and returns a real-time risk assessment for liver disease.
    """)

with col2:
    st.header("Patient Data Entry Form")
    st.markdown("Enter patient clinical metrics below to predict the likelihood of liver disease.")

    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)
        age = row1_col1.number_input("Age", min_value=1, max_value=120, value=20)
        gender_selection = row1_col2.selectbox("Gender", ["Female", "Male"])
        
        row2_col1, row2_col2 = st.columns(2)
        tb = row2_col1.number_input("Total Bilirubin", min_value=0.0, format="%.2f", value=1.21)
        db = row2_col2.number_input("Direct Bilirubin", min_value=0.0, format="%.2f", value=0.19)
        
        row3_col1, row3_col2 = st.columns(2)
        ap = row3_col1.number_input("Alkaline Phosphotase", min_value=0.0, format="%.2f", value=150.14)
        sgpt = row3_col2.number_input("Alamine Aminotransferase (SGPT)", min_value=0.0, format="%.2f", value=25.07)
        
        row4_col1, row4_col2 = st.columns(2)
        sgot = row4_col1.number_input("Aspartate Aminotransferase (SGOT)", min_value=0.0, format="%.2f", value=24.90)
        tp = row4_col2.number_input("Total Proteins", min_value=0.0, format="%.2f", value=6.85)
        
        row5_col1, row5_col2 = st.columns(2)
        alb = row5_col1.number_input("Albumin", min_value=0.0, format="%.2f", value=3.30)
        agr = row5_col2.number_input("Albumin/Globulin Ratio", min_value=0.0, format="%.2f", value=0.95)

        submitted = st.form_submit_button("Run Diagnostics", use_container_width=True)

    if submitted:

        gender_numeric = 1 if gender_selection == "Male" else 0
        

        input_features = np.array([[age, gender_numeric, tb, db, ap, sgpt, sgot, tp, alb, agr]])

        probabilities = model.predict_proba(input_features)[0]
        low_risk_prob = probabilities[0] * 100
        high_risk_prob = probabilities[1] * 100

        prediction = 1 if high_risk_prob >= 50 else 0

        st.divider()
        if prediction == 1:
            st.error(f"**High Risk Detected:** The model indicates that the patient is likely suffering from liver disease.")
            st.metric(label="Model Confidence Score (High Risk)", value=f"{high_risk_prob:.1f}%")
        else:
            st.success(f"**Low Risk:** The model does not detect strong signs of liver disease based on the provided metrics.")
            st.metric(label="Model Confidence Score (Low Risk)", value=f"{low_risk_prob:.1f}%")

st.divider()
st.caption("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
