# Campus Facility Reservation Monitoring (Streamlit)

A Streamlit adaptation of the submitted SAD Data Mining midterm script, now using the supplied `facility_reservations.csv` dataset.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

## Dataset
The supplied CSV is included in this project and loads by default. It contains 401 rows and 16 columns. The app maps `User_Category` to `User_Type` and `Lead_Time_Days` to `Days_In_Advance` so it can use the original model's feature names. The expected source columns also include `Reservation_ID`, `Facility_Type`, `Participants`, `Usage_Status`, `Actual_Attendance`, and `Cancellation_Days_Before`.

## Model
The original script did not include a separately saved model artifact. This project preserves the target (`Reservation_Outcome`), predictors (`User_Type`, `Facility_Type`, `Days_In_Advance`, `Participants`, `Lead_Time_Category`), 80/20 stratified split, random state 42, and Decision Tree depth 4. It saves a fitted preprocessing + model pipeline to `models/trained_model.joblib` on first run and loads it on later runs. The app uses a scikit-learn `OneHotEncoder` pipeline so new prediction inputs are encoded consistently.

## Deploy to Streamlit Community Cloud
Push the complete project folder to GitHub, then create an app at https://share.streamlit.io using `app.py` as the entry point. Keep the CSV in the repository if it is authorized for deployment; otherwise use synthetic data or a private/authorized deployment.

## Limitations
- Predictions are decision support, not automatic authorization to reject a reservation.
- The uploaded CSV is cleaned for analysis; uploading a different CSV does not automatically retrain the saved model.
- Model performance may differ from the original notebook because the supplied dataset schema and records differ from the synthetic fallback in the original script.
