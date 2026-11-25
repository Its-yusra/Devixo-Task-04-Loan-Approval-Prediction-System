
import streamlit as st
import pandas as pd
import joblib
import os

# Import custom transformer used during model training
from outlier_copper import OutlierCapper
import __main__

# Make custom transformer available to Joblib
__main__.OutlierCapper = OutlierCapper


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)


# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = "models/best_loan_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🏦 Loan Approval Prediction System")

st.write(
    "Enter the applicant's information below "
    "to predict the loan approval status."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. "
        "Please upload best_loan_model.joblib "
        "inside the models folder."
    )
    st.stop()

try:
    model = load_model()
    st.success("Machine Learning Model Loaded Successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ----------------------------------
# APPLICANT INPUT FORM
# ----------------------------------

st.subheader("Applicant Information")

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
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
        "Predict Loan Approval",
        use_container_width=True
    )


# ----------------------------------
# MAKE PREDICTION
# ----------------------------------

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

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("Loan Status: Approved")

        else:

            st.error("Loan Status: Not Approved")

        st.write(
            f"Approved Probability: {probabilities[1] * 100:.2f}%"
        )

        st.write(
            f"Not Approved Probability: {probabilities[0] * 100:.2f}%"
        )

        st.subheader("Applicant Summary")

        st.dataframe(input_data, use_container_width=True)

        st.caption(
            "Disclaimer: This is an educational machine learning "
            "application. Its predictions are not actual bank "
            "lending decisions."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")
