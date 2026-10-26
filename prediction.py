import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trained_model.joblib")
PREDICTOR_COLS = ["User_Type", "Facility_Type", "Days_In_Advance", "Participants", "Lead_Time_Category"]
TARGET_COL = "Reservation_Outcome"

def train_or_load_model(df):
    """Load the saved fitted pipeline if present; otherwise train once and save using original model settings."""
    if os.path.exists(MODEL_PATH):
        try:
            payload = joblib.load(MODEL_PATH)
            return payload["model"], payload["feature_columns"], payload["X_test"], payload["y_test"], payload["y_pred"]
        except Exception:
            pass
    X = df[PREDICTOR_COLS].copy()
    y = df[TARGET_COL].copy()
    categorical = ["User_Type", "Facility_Type", "Lead_Time_Category"]
    numeric = ["Days_In_Advance", "Participants"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    feature_columns = PREDICTOR_COLS
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns,
                 "X_test": X_test, "y_test": y_test, "y_pred": y_pred}, MODEL_PATH)
    return model, feature_columns, X_test, y_test, y_pred

def predict_one(model, feature_columns, user_type, facility_type, days, participants):
    lead = "Last Minute" if days <= 2 else "Short Lead" if days <= 7 else "Medium Lead" if days <= 14 else "Early Booking"
    row = pd.DataFrame([{
        "User_Type": user_type,
        "Facility_Type": facility_type,
        "Days_In_Advance": days,
        "Participants": participants,
        "Lead_Time_Category": lead
    }], columns=feature_columns)
    prediction = model.predict(row)[0]
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(row)[0]))
    return prediction, probabilities

def evaluate_model(y_test, y_pred):
    labels = ["Completed", "Not Completed"]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    return pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Not Completed)", "Recall (Not Completed)", "F1-score (Not Completed)"],
        "Score": [
            round(accuracy_score(y_test, y_pred), 4),
            round(precision_score(y_test, y_pred, pos_label="Not Completed", zero_division=0), 4),
            round(recall_score(y_test, y_pred, pos_label="Not Completed", zero_division=0), 4),
            round(f1_score(y_test, y_pred, pos_label="Not Completed", zero_division=0), 4)
        ]
    })

def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred, labels=["Completed", "Not Completed"])

def get_classification_report(y_test, y_pred):
    return classification_report(y_test, y_pred, zero_division=0, output_dict=True)
