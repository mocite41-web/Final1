import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules.data_processing import load_default_data, clean_data, normalise_schema, REQUIRED_COLUMNS
from modules.exploratory_analysis import show_eda
from modules.prediction import train_or_load_model, predict_one, evaluate_model
from modules.dashboard import show_dashboard
from modules.system_design import show_system_design

st.set_page_config(page_title="Facility Reservation Monitoring", page_icon="🏫", layout="wide")

st.title("🏫 Campus Facility Reservation Monitoring")
st.caption("Data mining decision-support application | Systems Analysis and Design")

pages = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
page = st.sidebar.selectbox("Navigate modules", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Model: Decision Tree Classifier (max_depth=4)")
st.sidebar.caption("Prediction target: Reservation_Outcome")

@st.cache_data
def get_default_data():
    raw = load_default_data()
    return raw

raw_default = get_default_data()
df_clean_default, cleaning_notes = clean_data(raw_default)
model, feature_columns, X_test, y_test, y_pred = train_or_load_model(df_clean_default)

if page == "Home":
    st.header("Home")
    st.write(
        "This application analyzes campus facility reservations and estimates whether a reservation "
        "will be completed or not completed. Predictions support facility administrators; they do not "
        "automatically approve or reject reservations."
    )
    a, b, c = st.columns(3)
    a.metric("Records in sample dataset", f"{len(df_clean_default):,}")
    b.metric("Facility types", df_clean_default["Facility_Type"].nunique())
    c.metric("Model", "Decision Tree")
    st.subheader("Workflow")
    st.markdown("1. Upload and validate reservation data.  2. Review data patterns and charts.  "
                "3. Enter pre-reservation details to generate a prediction.  "
                "4. A facility administrator reviews the result and records the final decision.")
    st.info("The supplied facility_reservations.csv is loaded as the default dataset. Its User_Category and "
            "Lead_Time_Days fields are mapped to the original model's User_Type and Days_In_Advance fields. "
            "The model uses Reservation_Outcome as its target and the pre-reservation predictors from your midterm script.")

elif page == "Dataset":
    st.header("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload facility reservation CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_raw_original = pd.read_csv(uploaded)
            df_raw = normalise_schema(df_raw_original)
            st.success(f"CSV loaded: {len(df_raw)} rows and {len(df_raw.columns)} columns.")
            missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
                st.write("Required schema:", REQUIRED_COLUMNS)
            elif df_raw.empty:
                st.warning("The uploaded file has no records.")
            else:
                df_clean, notes = clean_data(df_raw)
                st.subheader("Data preview")
                st.dataframe(df_raw.head(20), use_container_width=True)
                x, y, z = st.columns(3)
                x.metric("Rows", len(df_raw))
                y.metric("Duplicate reservation IDs", int(df_raw["Reservation_ID"].duplicated().sum()))
                z.metric("Missing cells", int(df_raw.isna().sum().sum()))
                st.subheader("Cleaning summary")
                for note in notes:
                    st.write("•", note)
                st.dataframe(df_clean.head(20), use_container_width=True)
                st.download_button("Download cleaned CSV", df_clean.to_csv(index=False).encode("utf-8"),
                                   "cleaned_facility_reservations.csv", "text/csv")
                st.caption("The uploaded data is validated and cleaned for inspection. Uploading a CSV does not silently retrain the saved model.")
        except Exception as e:
            st.error(f"Could not read this CSV. Please upload a valid CSV file. Details: {e}")
    else:
        st.write("Upload a CSV with the required reservation columns. A sample dataset is shown below.")
        st.dataframe(df_clean_default.head(20), use_container_width=True)
        st.write("Shape:", df_clean_default.shape)
        st.subheader("Variable types")
        st.dataframe(pd.DataFrame({"Column": df_clean_default.dtypes.index,
                                   "Data type": df_clean_default.dtypes.astype(str).values}))
        st.subheader("Summary statistics")
        st.dataframe(df_clean_default.describe(include="all").transpose(), use_container_width=True)
        st.download_button("Download sample cleaned CSV",
                           df_clean_default.to_csv(index=False).encode("utf-8"),
                           "cleaned_facility_reservations.csv", "text/csv")

elif page == "Data Analysis":
    st.header("Exploratory Data Analysis")
    show_eda(df_clean_default)

elif page == "Prediction":
    st.header("Reservation Outcome Prediction")
    st.write("Enter information available before the reservation takes place.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_type = st.selectbox("User type", ["Student", "Faculty", "Staff", "External"])
            facility_type = st.selectbox("Facility type", ["Auditorium", "Lab", "Conference Room", "Sports Hall"])
            days = st.number_input("Days in advance", min_value=0, max_value=365, value=7, step=1)
        with col2:
            participants = st.number_input("Expected participants", min_value=1, max_value=10000, value=30, step=1)
            st.caption("The model does not use actual attendance or cancellation timing because these are known only after booking.")
        submitted = st.form_submit_button("Predict reservation outcome", type="primary")
    if submitted:
        result, probabilities = predict_one(model, feature_columns, user_type, facility_type, int(days), int(participants))
        st.subheader("Prediction result")
        if result == "Not Completed":
            st.warning("Predicted outcome: Not Completed")
        else:
            st.success("Predicted outcome: Completed")
        if probabilities:
            st.write("Model class scores:")
            st.dataframe(pd.DataFrame([{"Outcome": k, "Score": round(v, 4)} for k, v in probabilities.items()]),
                         use_container_width=True)
        st.caption("Model version: DecisionTreeClassifier(max_depth=4), random_state=42. "
                   f"Prediction timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                   "Scores are model estimates, not guarantees. An authorized administrator must review context before acting.")

    st.subheader("Held-out test performance")
    st.dataframe(evaluate_model(y_test, y_pred), use_container_width=True)

elif page == "Dashboard":
    st.header("Decision-Support Dashboard")
    show_dashboard(df_clean_default, model, feature_columns, y_test, y_pred)

elif page == "System Design":
    st.header("SAD Integration")
    show_system_design()
