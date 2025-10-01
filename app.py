# app.py  (place at repo root OR in /app; works for both)
import os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Approval Prediction", page_icon="✅")
st.title("🏦 Loan Approval Prediction")

# --------- helpers ---------
def candidate_roots():
    here = os.path.dirname(__file__)
    return [
        here,                              # repo root if app.py at root
        os.path.join(here, ".."),          # parent
        os.path.join(here, "..", ".."),    # grandparent
        os.path.join(here, "app"),         # if app.py at root but models under /app/../models
        os.path.join(here, "..", "app"),   # if app structure exists
    ]

def find_models_dir():
    for base in candidate_roots():
        base = os.path.abspath(base)
        m = os.path.join(base, "models")
        if os.path.isdir(m):
            return m, base
    # fallback to current dir/models
    base = os.path.abspath(os.path.dirname(__file__))
    m = os.path.join(base, "models")
    os.makedirs(m, exist_ok=True)
    return m, base

MODELS_DIR, BASE = find_models_dir()
SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")
MODEL_PATH  = os.path.join(MODELS_DIR, "loan_model.pkl")

# minimal trainer (runs if files missing)
def train_if_needed():
    if os.path.exists(SCHEMA_PATH) and os.path.exists(MODEL_PATH):
        return
    st.warning("Model files not found — training a small model now (one-time).")

    # Generate a small synthetic dataset
    rng = np.random.default_rng(42)
    n = 2000
    genders = rng.choice(["Male", "Female"], size=n)
    married = rng.choice(["Yes", "No"], size=n)
    dependents = rng.choice(["0","1","2","3+"], size=n)
    education = rng.choice(["Graduate","Not Graduate"], size=n)
    self_emp = rng.choice(["Yes","No"], size=n)
    applicant_income = rng.integers(20000,100000,size=n)
    co_income = rng.integers(0,30000,size=n)
    loan_amt = rng.integers(50000,500000,size=n)
    term = rng.choice([120,180,240,300,360], size=n)
    credit = rng.choice([1.0,0.0], size=n, p=[0.8,0.2])
    prop = rng.choice(["Urban","Semiurban","Rural"], size=n)
    age = rng.integers(21,60,size=n)
    cibil = rng.integers(300,900,size=n)
    existing = rng.integers(0,3,size=n)

    emi_to_income = (loan_amt/term) / (applicant_income + co_income + 1)
    emi_to_income = np.clip(emi_to_income, 0.01, 1.5)
    approved = ((credit==1.0)&(cibil>650)&(emi_to_income<0.5)).astype(int)

    df = pd.DataFrame({
        "Gender":genders,"Married":married,"Dependents":dependents,"Education":education,
        "Self_Employed":self_emp,"ApplicantIncome":applicant_income,"CoapplicantIncome":co_income,
        "LoanAmount":loan_amt,"Loan_Amount_Term":term,"Credit_History":credit,
        "Property_Area":prop,"Age":age,"CIBIL_Score":cibil,"Existing_Loans":existing,
        "EMI_to_Income":np.round(emi_to_income,3),"Loan_Status":approved
    })

    schema = {
        "categorical": ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"],
        "numeric": ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History",
                    "Age","CIBIL_Score","Existing_Loans","EMI_to_Income"],
        "target": "Loan_Status",
        "classes": {"0":"Rejected","1":"Approved"}
    }

    cat = schema["categorical"]; num = schema["numeric"]; y = df[schema["target"]].values
    X = df[cat + num]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    model = Pipeline([("preprocess", preprocess), ("clf", LogisticRegression(max_iter=1000))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(Xtr, ytr)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SCHEMA_PATH, "w") as f: json.dump(schema, f, indent=2)
    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)

train_if_needed()

# ---- load artifacts ----
with open(SCHEMA_PATH, "r") as f:
    SCHEMA = json.load(f)
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

# ---- UI ----
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
        "Gender": Gender, "Married": Married, "Dependents": Dependents, "Education": Education,
        "Self_Employed": Self_Employed, "Property_Area": Property_Area,
        "ApplicantIncome": ApplicantIncome, "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount, "Loan_Amount_Term": int(Loan_Amount_Term),
        "Credit_History": float(Credit_History), "Age": int(Age),
        "CIBIL_Score": int(CIBIL_Score), "Existing_Loans": int(Existing_Loans),
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
