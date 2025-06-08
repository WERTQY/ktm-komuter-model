import streamlit as st
import pandas as pd
import numpy as np
import holidays
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load

# -------------------- Transformers --------------------
class HolidayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, country="MY", subdiv=None):
        # Use python-holidays for dynamic holiday lookup
        self.hol = holidays.CountryHoliday(country, subdiv=subdiv)
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # flag if date is holiday
        X["is_holiday"] = X['date'].dt.date.apply(lambda d: d in self.hol)
        return X

class GeoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, station_coords: dict):
        # station_coords: {name: (lat, lon)}
        self.coords = station_coords
        # define central hub coordinate
        self.center = self.coords.get("KL Sentral")
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # straight-line distance
        X['straight_km'] = X.apply(
            lambda r: geodesic(self.coords[r.origin], self.coords[r.destination]).km
            if r.origin in self.coords and r.destination in self.coords else np.nan,
            axis=1
        )
        # distance to center
        if self.center:
            X['orig_center_km'] = X.origin.map(lambda s: geodesic(self.coords[s], self.center).km if s in self.coords else np.nan)
            X['dest_center_km'] = X.destination.map(lambda s: geodesic(self.coords[s], self.center).km if s in self.coords else np.nan)
        return X

class TimeCyclicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # hour cyclic
        X['hour_sin'] = np.sin(2 * np.pi * X.hour / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X.hour / 24)
        # day-of-week cyclic
        dow = X['date'].dt.dayofweek
        X['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        X['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        return X

class FeaturePipeline:
    def __init__(self, station_coords: dict, country="MY", subdiv=None):
        self.steps = [
            HolidayTransformer(country=country, subdiv=subdiv),
            GeoTransformer(station_coords),
            TimeCyclicTransformer()
            # add more transformers here as needed
        ]
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for step in self.steps:
            X = step.transform(X)
        # select features for model
        features = [
            'straight_km', 'orig_center_km', 'dest_center_km',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_holiday'
        ]
        return X[features]

# -------------------- Load model & coords --------------------
# Replace paths with your actual files
@st.cache(allow_output_mutation=True)
def load_model_and_pipeline():
    """
    Load trained model and station coordinates mapping.
    Station coordinates can be provided as a joblib dump of a dict or as a CSV file.
    """
    # Load model
    model = load("model.joblib")
    # Load station coordinates mapping
    try:
        # Expect a dict saved via joblib: {station_name: (lat, lon)}
        station_coords = load("station_coords.joblib")
    except Exception:
        # Fallback: load from CSV with columns station, lat, lon
        df_coor = pd.read_csv("station_coords.csv")
        station_coords = {
            row['station']: (row['lat'], row['lon'])
            for _, row in df_coor.iterrows()
        }
    # Build feature pipeline
    pipeline = FeaturePipeline(station_coords)
    return model, pipeline

# -------------------- Streamlit UI --------------------
st.title("🚆 KTM Komuter Ridership Demo")

# 1. Input: upload CSV or manual
upload = st.file_uploader("Upload raw ridership CSV (with columns: origin, destination, date, hour)", type="csv")
if upload:
    df_raw = pd.read_csv(upload, parse_dates=["date"])  # expect columns: origin,destination,date,hour
else:
    st.info("Or enter a single record manually below:")
    origin = st.text_input("Origin station")
    dest = st.text_input("Destination station")
    date = st.date_input("Date")
    hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=8)
    if origin and dest:
        df_raw = pd.DataFrame([{"origin": origin, "destination": dest, "date": pd.to_datetime(date), "hour": hour}])
    else:
        df_raw = pd.DataFrame([])

# 2. Holiday library option
use_lib = st.checkbox("Use dynamic holiday lookup (python-holidays)", value=True)
# subdiv selection if dynamic
subdiv = None
if use_lib:
    subdiv = st.selectbox("Select state/subdivision for holidays", [None] + sorted(holidays.Malaysia().subdivisions))

# 3. Predict
if not df_raw.empty and st.button("Run prediction"):
    model, pipeline = load_model_and_pipeline()
    # transform
    df_feat = pipeline.transform(df_raw)
    preds = model.predict(df_feat)
    df_raw["predicted_ridership"] = preds
    st.table(df_raw)
else:
    st.write("Awaiting input...")

st.markdown("---\n*Built with Streamlit* and python-holidays*")
