import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_eda(df):
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    st.subheader("Class distribution")
    st.bar_chart(df["Reservation_Outcome"].value_counts())
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reservations by facility type**")
        counts = df["Facility_Type"].value_counts()
        st.bar_chart(counts)
    with c2:
        st.markdown("**Outcome by lead-time category (%)**")
        table = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index") * 100
        st.bar_chart(table)
    st.markdown("**Outcome by user type (%)**")
    user_table = pd.crosstab(df["User_Type"], df["Reservation_Outcome"], normalize="index") * 100
    st.bar_chart(user_table)
    st.subheader("Interactive chart filters")
    facility = st.selectbox("Filter by facility", ["All"] + sorted(df["Facility_Type"].dropna().unique().tolist()))
    filtered = df if facility == "All" else df[df["Facility_Type"] == facility]
    st.dataframe(filtered[["Reservation_ID", "User_Type", "Facility_Type", "Days_In_Advance",
                           "Lead_Time_Category", "Usage_Status", "Reservation_Outcome"]].head(100),
                 use_container_width=True)
