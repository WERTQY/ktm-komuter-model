import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta

# --- 1) Load your trained pipeline/model ---
@st.cache(allow_output_mutation=True)
def load_pipeline(path: str = 'pipeline.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

pipeline = load_pipeline('pipeline.pkl')

# --- 2) Load historical ridership for recursion ---
MODEL_CUTOFF = datetime(2025, 5, 12).date()
START_DATE = MODEL_CUTOFF + timedelta(days=1)

@st.cache
def load_history(path: str = 'full_history.parquet') -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Keep only up to cutoff date
    df = df[df['date'].dt.date <= MODEL_CUTOFF]
    return df

history_df = load_history()

# --- 3) UI: user inputs ---
st.title('KTM Komuter Ridership Prediction')
stations = sorted(history_df['origin'].unique())

origin = st.selectbox('Origin Station', stations)
destination = st.selectbox('Destination Station', stations)

selected_date = st.date_input(
    'Select date for prediction',
    min_value=START_DATE,
    max_value=datetime.now().date()
)

hour = st.slider('Select hour of day', 0, 23, 8)

# --- 4) Recursive / single forecast functions ---
def recursive_forecast(orig: str, dest: str, end_date: datetime.date, hr: int) -> pd.DataFrame:
    # Filter history for this OD pair
    hist = history_df[
        (history_df['origin'] == orig) &
        (history_df['destination'] == dest)
    ].copy()
    hist.set_index('date', inplace=True)

    out = []
    current = START_DATE
    while current <= end_date:
        # Prepare feature row
        df_feat = pd.DataFrame({
            'origin': [orig],
            'destination': [dest],
            'date': [pd.to_datetime(current)],
            'hour': [hr]
        })
        # Optionally: compute any lag features here from hist
        # e.g., df_feat['lag_1'] = hist.loc[current - timedelta(days=1), 'ridership'] if (current - timedelta(days=1)) in hist.index else 0
        
        yhat = pipeline.predict(df_feat)[0]
        out.append({'date': current, 'predicted_ridership': int(yhat)})

        # Append prediction to hist for next iteration
        hist.loc[pd.to_datetime(current), 'ridership'] = yhat
        current += timedelta(days=1)

    return pd.DataFrame(out)

# --- 5) Buttons & display ---
if st.button('Predict Series'):
    df_res = recursive_forecast(origin, destination, selected_date, hour)
    st.subheader('Recursive Forecast Results')
    st.dataframe(df_res, use_container_width=True)
elif st.button('Predict Single'):
    df_single = pd.DataFrame({
        'origin': [origin],
        'destination': [destination],
        'date': [pd.to_datetime(selected_date)],
        'hour': [hour]
    })
    yhat = pipeline.predict(df_single)[0]
    st.write(f"Predicted ridership on {selected_date} at {hour}:00 → {int(yhat)}")

# Note: extend recursive_forecast to include lag/window features as needed for your model
