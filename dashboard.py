import streamlit as st
import pandas as pd
from modules.prediction import evaluate_model

def show_dashboard(df, model, feature_columns, y_test, y_pred):
    completed = int((df["Reservation_Outcome"] == "Completed").sum())
    not_completed = int((df["Reservation_Outcome"] == "Not Completed").sum())
    completion_rate = 100 * completed / len(df) if len(df) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reservations", f"{len(df):,}")
    c2.metric("Completed", f"{completed:,}")
    c3.metric("Not completed", f"{not_completed:,}")
    c4.metric("Completion rate", f"{completion_rate:.1f}%")
    st.subheader("Review queue")
    queue = df[df["Reservation_Outcome"] == "Not Completed"].copy()
    queue["Review reason"] = queue["Usage_Status"].map({
        "Cancelled": "Reservation cancelled",
        "No-Show": "Recorded no-show"
    }).fillna("Outcome needs review")
    st.dataframe(queue[["Reservation_ID", "User_Type", "Facility_Type", "Days_In_Advance",
                        "Usage_Status", "Review reason"]].head(100), use_container_width=True)
    st.subheader("Held-out model metrics")
    st.dataframe(evaluate_model(y_test, y_pred), use_container_width=True)
    st.markdown("**Recommended human follow-up**")
    st.markdown("- Verify booking details and facility availability.\n- Contact the requester for confirmation where appropriate.\n"
                "- Review context before making a decision.\n- Record the authorized administrator's final decision.")
    st.warning("Predictions and observed outcomes are decision-support only. Do not automatically reject reservations based on model output.")
