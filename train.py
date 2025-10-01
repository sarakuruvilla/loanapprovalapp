# src/train.py
import os, json, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAVE_SMOTE = True
except Exception:
    HAVE_SMOTE = False

# ----- Paths -----
BASE = os.path.dirname(os.path.dirname(__file__))           # project root
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH   = os.path.join(DATA_DIR,   "loan_data.csv")
SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")
MODEL_PATH  = os.path.join(MODELS_DIR, "loan_model.pkl")
METRICS_PATH= os.path.join(MODELS_DIR, "metrics.json")

# ----- Dataset generator (used if CSV missing) -----
def generate_synthetic_loan_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    genders = rng.choice(["Male","Female"], size=n, p=[0.6,0.4])
    married = rng.choice(["Yes","No"], size=n, p=[0.7,0.3])
    dependents = rng.choice(["0","1","2","3+"], size=n, p=[0.55,0.2,0.15,0.10])
    education = rng.choice(["Graduate","Not Graduate"], size=n, p=[0.7,0.3])
    self_emp = rng.choice(["Yes","No"], size=n, p=[0.2,0.8])

    applicant_income  = rng.normal(45000,20000,size=n).clip(5000,200000).round()
    coapplicant_income= (rng.exponential(12000,size=n)*rng.choice([0,1],p=[0.5,0.5],size=n)).round()
    loan_amount       = rng.normal(250000,125000,size=n).clip(25000,1500000).round()
    term              = rng.choice([120,180,240,300,360], size=n, p=[0.05,0.1,0.2,0.25,0.4])
    credit            = rng.choice([1.0,0.0], size=n, p=[0.8,0.2])
    prop              = rng.choice(["Urban","Semiurban","Rural"], size=n, p=[0.4,0.35,0.25])
    age               = rng.integers(21,62,size=n)
    cibil             = rng.normal(720,60,size=n).clip(300,900).round()
    existing_loans    = rng.choice([0,1,2,3], size=n, p=[0.55,0.3,0.1,0.05])

    emi_to_income = (loan_amount/np.maximum(term,1)) / np.maximum(applicant_income+coapplicant_income,1)
    emi_to_income = (emi_to_income * rng.normal(1.0,0.2,size=n)).clip(0.01,1.5)

    # heuristic approval probability
    p = (0.45
         + 0.25*(credit==1.0)
         + 0.10*(cibil/900)
         + 0.08*(education=="Graduate")
         + 0.05*(prop=="Urban")
         - 0.25*emi_to_income
         - 0.05*existing_loans/3.0
         + 0.05*(applicant_income>50000))
    p = np.clip(p, 0.02, 0.98)
    status = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "Gender":genders,"Married":married,"Dependents":dependents,"Education":education,
        "Self_Employed":self_emp,"ApplicantIncome":applicant_income.astype(int),
        "CoapplicantIncome":coapplicant_income.astype(int),"LoanAmount":loan_amount.astype(int),
        "Loan_Amount_Term":term.astype(int),"Credit_History":credit,"Property_Area":prop,
        "Age":age.astype(int),"CIBIL_Score":cibil.astype(int),"Existing_Loans":existing_loans.astype(int),
        "EMI_to_Income":np.round(emi_to_income,3),"Loan_Status":status
    })
    return df

# Create CSV if missing
if not os.path.exists(DATA_PATH):
    df_new = generate_synthetic_loan_data()
    df_new.to_csv(DATA_PATH, index=False)
    print(f"🆕 Created dataset: {DATA_PATH}  shape={df_new.shape}")

# ----- Load data -----
df = pd.read_csv(DATA_PATH)

# Self-heal columns if any are missing
defaults = {
    "CoapplicantIncome": 0, "Existing_Loans": 0, "Loan_Amount_Term": 360,
    "Credit_History": 1.0, "Age": 30, "CIBIL_Score": 720
}
for c, v in defaults.items():
    if c not in df.columns: df[c] = v
if "EMI_to_Income" not in df.columns:
    denom = (df.get("ApplicantIncome",0).fillna(0) + df.get("CoapplicantIncome",0).fillna(0)).clip(lower=1)
    term  = df.get("Loan_Amount_Term",360).replace(0,360)
    emi_monthly = df.get("LoanAmount",0).fillna(0) / term
    df["EMI_to_Income"] = (emi_monthly/denom).clip(0.01,1.5)

# ----- Schema -----
schema = {
    "categorical": ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"],
    "numeric": ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History",
                "Age","CIBIL_Score","Existing_Loans","EMI_to_Income"],
    "target": "Loan_Status",
    "classes": {"0":"Rejected","1":"Approved"}
}
with open(SCHEMA_PATH,"w") as f: json.dump(schema, f, indent=2)

cat_features = schema["categorical"]
num_features = schema["numeric"]
target = schema["target"]

X = df[cat_features + num_features]
y = df[target].values

# ----- Preprocess + model -----
numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())])
categorical_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([("num", numeric_tf, num_features),
                                ("cat", categorical_tf, cat_features)])

classes = np.unique(y)
cw = {int(c): w for c, w in zip(classes,
                                compute_class_weight("balanced", classes=classes, y=y))}
clf = LogisticRegression(max_iter=1000, class_weight=cw)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipeline = Pipeline([("preprocess", preprocess), ("clf", clf)])
cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy").mean()
cv_f1  = cross_val_score(pipeline, X, y, cv=cv, scoring="f1").mean()
cv_roc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
if HAVE_SMOTE:
    model = ImbPipeline([("preprocess", preprocess), ("smote", SMOTE(random_state=42)), ("clf", clf)])
else:
    model = Pipeline([("preprocess", preprocess), ("clf", clf)])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "cv_accuracy_mean": float(cv_acc),
    "cv_f1_mean": float(cv_f1),
    "cv_roc_auc_mean": float(cv_roc),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_f1": float(f1_score(y_test, y_pred)),
    "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
}
with open(METRICS_PATH,"w") as f: json.dump(metrics, f, indent=2)
with open(MODEL_PATH,"wb") as f: pickle.dump(model, f)

print("✅ Training complete")
print("Saved:", MODEL_PATH)
print("Saved:", METRICS_PATH)
