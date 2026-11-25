
import streamlit as st
import pandas as pd
import joblib
import os
from outlier_capper import OutlierCapper

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)

# Load trained model
MODEL_PATH = "models/best_loan_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Application title
st.title("🏦 Loan Approval Prediction System")

st.write(
    "Enter the applicant's information below "
    "to predict the loan approval status."
)

# Check model availability
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please add the model file.")
    st.stop()

model = load_model()

# Input form
st.subheader("Applicant Information")

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox(
            "Dependents", ["0", "1", "2", "3+"]
        )
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox(
            "Self Employed", ["Yes", "No"]
        )
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

    with col2:
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands)",
            min_value=0.0,
            value=150.0
        )
        loan_term = st.selectbox(
            "Loan Amount Term (days)",
            [360.0, 180.0, 240.0, 120.0, 300.0, 480.0, 60.0]
        )
        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0]
        )
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    submitted = st.form_submit_button(
        "Predict Loan Approval"
    )

# Prediction
if submitted:

    total_income = applicant_income + coapplicant_income

    if total_income > 0:
        loan_to_income_ratio = loan_amount / total_income
    else:
        loan_to_income_ratio = 0.0

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Total_Income": total_income,
        "Loan_to_Income_Ratio": loan_to_income_ratio
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Loan Status: Approved")
    else:
        st.error("Loan Status: Not Approved")

    st.write(
        f"Not Approved Probability: {probability[0] * 100:.2f}%"
    )
    st.write(
        f"Approved Probability: {probability[1] * 100:.2f}%"
    )

    st.caption(
        "This application is an educational machine learning demo. "
        "Its prediction is not an actual bank lending decision."
    )
