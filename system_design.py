import streamlit as st

def show_system_design():
    st.subheader("Actors")
    st.markdown("- **Requester:** submits a facility reservation and confirms attendance details.\n"
                "- **Facility Administrator:** reviews reservation risks, checks availability, and records the final decision.\n"
                "- **System Administrator:** manages the application, dataset access, and model version.")
    st.subheader("Five functional requirements")
    requirements = [
        ("FR-DM-01", "Flag reservations made with fewer than 3 days' lead time for confirmation review."),
        ("FR-DM-02", "Allow an authorized administrator to review student and external-user bookings for high-demand facilities."),
        ("FR-DM-03", "Display a dashboard of upcoming reservation risk and utilization indicators."),
        ("FR-DM-04", "Validate uploaded CSV files and report missing required columns before analysis."),
        ("FR-DM-05", "Show model predictions and evaluation metrics while requiring an authorized human to make the final decision.")
    ]
    for code, text in requirements:
        st.write(f"**{code}** — {text}")
    st.subheader("Three non-functional requirements")
    st.markdown("- **Security:** use authorized/synthetic data and restrict access to reservation records.\n"
                "- **Usability:** show clear validation messages, readable tables, and understandable prediction explanations.\n"
                "- **Reliability:** handle empty files, missing columns, and unexpected input without crashing.")
    st.subheader("Use case summary")
    st.markdown("""
    ```text
    Requester  ---> Submit reservation / Confirm booking
                         |
                         v
    Facility Administrator ---> View dashboard / Review risk / Record final decision
                         ^
                         |
    System Administrator ---> Manage dataset access / Review model version
    ```
    """)
    st.subheader("Analytical process flow")
    st.markdown("CSV upload → schema validation → cleaning and feature engineering → exploratory analysis → "
                "Decision Tree prediction → administrator review → final human decision.")
    st.info("The original midterm script only stated three evidence-based functional requirements. FR-DM-04 and FR-DM-05, plus the non-functional requirements, are added to meet this Streamlit exercise's submission requirements.")
