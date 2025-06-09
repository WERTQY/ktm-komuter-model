import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# --- 1) Load your trained pipeline/model ---
@st.cache(allow_output_mutation=True)
def load_pipeline(path: str = 'pipeline.pkl'):
    """Load a sklearn Pipeline that includes preprocessing & model."""
    with open(path, 'rb') as f:
        return pickle.load(f)

pipeline = load_pipeline('pipeline.pkl')

# --- 2) Define the forecasting range ---
MODEL_CUTOFF = datetime(2025, 5, 12).date()
START_DATE = MODEL_CUTOFF + pd.Timedelta(days=1)

# --- 3) UI: user inputs ---
st.title('KTM Komuter Ridership Prediction')

# Replace this list with your actual station names
stations = [
    'Tanjung Malim', 'Kuala Kubu Bharu', 'Rasa', 'Batang Kali', 'Serendah',
    'Rawang', 'Kuang', 'Sungai Buloh', 'Kepong Sentral', 'Kepong', 'Segambut',
    'Putra', 'Bank Negara', 'Kuala Lumpur', 'KL Sentral', 'Abdullah Hukum',
    'Angkasapuri', 'Pantai Dalam', 'Petaling', 'Jalan Templer',
    'Kampung Dato Harun', 'Seri Setia', 'Setia Jaya', 'Subang Jaya',
    'Batu Tiga', 'Shah Alam', 'Padang Jawa', 'Bukit Badak', 'Klang',
    'Teluk Pulai', 'Teluk Gadong', 'Kampung Raja Uda', 'Jalan Kastam',
    'Pelabuhan Klang', 'Batu Caves', 'Taman Wahyu', 'Kampung Batu',
    'Batu Kentonmen', 'Sentul', 'MidValley', 'Seputeh', 'Salak Selatan',
    'Bandar Tasek Selatan', 'Serdang', 'Kajang', 'Kajang 2', 'UKM',
    'Bangi', 'Batang Benar', 'Nilai', 'Labu', 'Tiroi', 'Seremban',
    'Senawang', 'Sungai Gadut', 'Rembau', 'Pulau Sebang'
]

origin = st.selectbox('Origin Station', stations)
destination = st.selectbox('Destination Station', stations)

selected_date = st.date_input(
    'Select date for prediction',
    min_value=START_DATE,
    max_value=datetime.now().date()
)

hour = st.slider('Select hour of day', 0, 23, 8)

# --- 4) Recursive / single forecast functions ---
def recursive_forecast(orig, dest, end_date, hr):
    """Generate predictions from START_DATE up to end_date (inclusive)."""
    dates = pd.date_range(START_DATE, end_date, freq='D')
    results = []
    for d in dates:
        # prepare input
        df = pd.DataFrame({
            'origin': [orig],
            'destination': [dest],
            'date': [d],
            'hour': [hr]
        })
        # ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        # predict
        yhat = pipeline.predict(df)[0]
        results.append({'date': d.date(), 'predicted_ridership': int(yhat)})
    return pd.DataFrame(results)

# --- 5) Buttons & display ---
if st.button('Predict Series'):
    df_res = recursive_forecast(origin, destination, selected_date, hour)
    st.subheader('Recursive Forecast Results')
    st.dataframe(df_res, use_container_width=True)
elif st.button('Predict Single'):
    df_single = pd.DataFrame({
        'origin': [origin],
        'destination': [destination],
        'date': [selected_date],
        'hour': [hour]
    })
    df_single['date'] = pd.to_datetime(df_single['date'])
    yhat = pipeline.predict(df_single)[0]
    st.write(f"Predicted ridership on {selected_date} at {hour}:00 → {int(yhat)}")
