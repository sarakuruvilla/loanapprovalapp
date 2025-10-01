# app/app.py
import streamlit as st
import pickle, json, os
import numpy as np

st.set_page_config(page_title="Loan Approval Demo", page_icon="✅")
st.title("🏦 Loan Approval Prediction")

BASE = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(BASE, "models", "feature_schema.json")) as f:
    SCHEMA = json.load(f)
with open(os.path.join(BASE, "models", "loan_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0","1","2","3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
        Property_Area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
        Age = st.number_input("Age", min_value=18, max_value=75, value=30, step=1)
        CIBIL_Score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=720, step=1)
    with col2:
        ApplicantIncome = st.number_input("Applicant Income (₹/month)", min_value=0, value=50000, step=1000)
        CoapplicantIncome = st.number_input("Coapplicant Income (₹/month)", min_value=0, value=0, step=1000)
        LoanAmount = st.number_input("Loan Amount (₹)", min_value=10000, value=300000, step=5000)
        Loan_Amount_Term = st.selectbox("Loan Term (months)", [120,180,240,300,360], index=4)
        Credit_History = st.selectbox("Credit History", [1.0, 0.0], index=0)
        Existing_Loans = st.number_input("Existing Loans (count)", min_value=0, max_value=5, value=0, step=1)
        EMI_to_Income = st.number_input("EMI to Income Ratio", min_value=0.01, max_value=1.5, value=0.25, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "Property_Area": Property_Area,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": int(Loan_Amount_Term),
        "Credit_History": float(Credit_History),
        "Age": int(Age),
        "CIBIL_Score": int(CIBIL_Score),
        "Existing_Loans": int(Existing_Loans),
        "EMI_to_Income": float(EMI_to_Income)
    }
    cols = SCHEMA["categorical"] + SCHEMA["numeric"]
    X = np.array([[payload[c] for c in cols]], dtype=object)
    proba = float(MODEL.predict_proba(X)[0,1])
    pred = int(proba >= 0.5)
    st.subheader("Result")
    st.metric("Approval Probability", f"{proba*100:.1f}%")
    st.write("Prediction:", "**Approved** ✅" if pred==1 else "**Rejected** ❌")
    with st.expander("See submitted details"):
        st.json(payload)
