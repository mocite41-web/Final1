import numpy as np
import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "Reservation_ID", "User_Type", "Facility_Type", "Days_In_Advance",
    "Participants", "Usage_Status", "Actual_Attendance", "Cancellation_Days_Before"
]
APP_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = APP_ROOT / "facility_reservations.csv"

def load_default_data():
    """Load the supplied facility reservations dataset; fall back to reproducible synthetic data."""
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        # Normalize this submitted dataset's column names to the original midterm model schema.
        aliases = {
            "User_Category": "User_Type",
            "Lead_Time_Days": "Days_In_Advance",
        }
        raw = raw.rename(columns={k: v for k, v in aliases.items() if k in raw.columns and v not in raw.columns})
        return raw
    np.random.seed(42)
    n = 400
    return pd.DataFrame({
        "Reservation_ID": [f"RES-{1000+i}" for i in range(n)],
        "User_Type": np.random.choice(["Student", "Faculty", "Staff", "External"], size=n,
                                      p=[0.5, 0.25, 0.15, 0.10]),
        "Facility_Type": np.random.choice(["Auditorium", "Lab", "Conference Room", "Sports Hall"], size=n),
        "Days_In_Advance": np.random.randint(-1, 30, size=n),
        "Participants": np.random.randint(5, 100, size=n),
        "Usage_Status": np.random.choice(["Used", "Cancelled", "No-Show"], size=n, p=[0.6, 0.25, 0.15]),
        "Actual_Attendance": np.random.randint(0, 100, size=n),
        "Cancellation_Days_Before": np.random.choice([np.nan, 1, 2, 5], size=n)
    })

def normalise_schema(df):
    df = df.copy()
    aliases = {"User_Category": "User_Type", "Lead_Time_Days": "Days_In_Advance"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    return df

def categorise_lead_time(days):
    if days <= 2:
        return "Last Minute"
    if days <= 7:
        return "Short Lead"
    if days <= 14:
        return "Medium Lead"
    return "Early Booking"

def clean_data(df):
    df = normalise_schema(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    original_rows = len(df)
    dupes = int(df["Reservation_ID"].duplicated().sum())
    df = df.drop_duplicates(subset=["Reservation_ID"]).copy()
    df["Days_In_Advance"] = pd.to_numeric(df["Days_In_Advance"], errors="coerce").clip(lower=0)
    median_days = df["Days_In_Advance"].median()
    df["Days_In_Advance"] = df["Days_In_Advance"].fillna(0 if pd.isna(median_days) else median_days)
    df["Participants"] = pd.to_numeric(df["Participants"], errors="coerce")
    median_participants = df["Participants"].median()
    df["Participants"] = df["Participants"].fillna(1 if pd.isna(median_participants) else median_participants)
    df["Actual_Attendance"] = pd.to_numeric(df["Actual_Attendance"], errors="coerce").fillna(0)
    df["Lead_Time_Category"] = df["Days_In_Advance"].apply(categorise_lead_time)
    df["Reservation_Outcome"] = df["Usage_Status"].apply(
        lambda x: "Completed" if str(x).strip().lower() == "used" else "Not Completed"
    )
    df["Attendance_Ratio"] = np.where(
        (df["Participants"].isna()) | (df["Participants"] == 0), 0.0,
        df["Actual_Attendance"] / df["Participants"]
    )
    notes = [
        f"Removed {dupes} duplicate Reservation_ID rows ({original_rows} original rows).",
        "Negative or missing lead-time values were clipped/filled using the median where available.",
        "Missing Participants values were filled using the median where available.",
        "Mapped User_Category to User_Type and Lead_Time_Days to Days_In_Advance where needed.",
        "Added Lead_Time_Category, Reservation_Outcome, and Attendance_Ratio."
    ]
    return df, notes
