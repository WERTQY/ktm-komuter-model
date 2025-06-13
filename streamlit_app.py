import streamlit as st
import pandas as pd
import pickle
from datetime import date

# --- 1) Load or build trained pipeline ---
import os
import subprocess

@st.cache_resource
def load_pipeline(path='pipeline.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Ensure pipeline exists; if not, build it
if not os.path.exists('pipeline.pkl'):
    with st.spinner("No pipeline found; training model pipeline…"):
        subprocess.run(["python", "pipeline.py", "--train"], check=True)
    load_pipeline.clear()

pipeline = load_pipeline()


# --- 2) Build UI for user inputs ---
st.subheader("Enter Trip Details")
origin = st.text_input(
    "Origin Station", 
    value="Sentul",
    help="Type a KTM station name, e.g. 'Sentul'"
)
destination = st.text_input(
    "Destination Station", 
    value="Kuala Lumpur",
    help="Type a KTM station name, e.g. 'Kuala Lumpur'"
)
# Date picker: no earlier than 2025-05-13
day = st.date_input(
    "Date (>= May 13, 2025)",
    value=date(2025, 5, 13),
    min_value=date(2025, 5, 13)
)
# Hour selector
hour = st.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12,
    help="Select hour (0–23)"
)


# Optionally rebuild pipeline from raw data
if st.button("Rebuild Model Pipeline"):
    import subprocess
    try:
        with st.spinner("Rebuilding pipeline…"):
            subprocess.run(["python", "pipeline.py", "--train"], check=True)
        load_pipeline.clear()
        pipeline = load_pipeline()
        st.success("🔄 Pipeline rebuilt successfully!")
    except Exception as e:
        st.error(f"Failed to rebuild pipeline: {e}")

# Inputs:) Trigger prediction ---
if st.button("Predict Ridership"):
    # Assemble a single-row DataFrame
    df_input = pd.DataFrame({
        'origin':      [origin],
        'destination': [destination],
        'date':        pd.to_datetime([day]),
        'hour':        [hour]
    })
    try:
        # Run through pipeline
        pred = pipeline.predict(df_input)[0]
        st.success(f"🚆 Estimated ridership: {int(round(pred))} passengers")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- 4) CLI fallback ---
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict ridership via CLI or launch Streamlit app if no args.")
    parser.add_argument('--origin',      type=str,   help='Origin station')
    parser.add_argument('--destination', type=str,   help='Destination station')
    parser.add_argument('--date',        type=str,   help='Date YYYY-MM-DD')
    parser.add_argument('--hour',        type=int,   choices=range(0,24), metavar='[0-23]', help='Hour of day')
    args = parser.parse_args()

    # If all CLI args are provided, run prediction
    if args.origin and args.destination and args.date is not None and args.hour is not None:
        df_input = pd.DataFrame({
            'origin':      [args.origin],
            'destination': [args.destination],
            'date':        pd.to_datetime([args.date]),
            'hour':        [args.hour]
        })
        try:
            pred = pipeline.predict(df_input)[0]
            print(f"Estimated ridership: {int(round(pred))}")
        except Exception as e:
            print(f"Prediction failed: {e}")
    else:
        # Launch Streamlit's UI
        from streamlit.web import cli as st_cli
        st_cli._main_run_clExplicitly()
