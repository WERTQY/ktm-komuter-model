import streamlit as st
import pandas as pd
import numpy as np
import holidays
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load

# -------------------- Transformers --------------------
class HolidayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, country="MY"):  # national holidays
        self.hol = holidays.CountryHoliday(country)
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # flag if date is holiday
        X["is_holiday"] = X['date'].dt.date.apply(lambda d: d in self.hol)
        # optional: get holiday name if needed
        X["holiday_name"] = X['date'].dt.date.map(lambda d: self.hol.get(d))
        return X

class GeoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, station_coords: dict):
        self.coords = station_coords
        self.center = self.coords.get("KL Sentral")
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['straight_km'] = X.apply(
            lambda r: geodesic(self.coords[r.origin], self.coords[r.destination]).km
            if r.origin in self.coords and r.destination in self.coords else np.nan,
            axis=1
        )
        if self.center:
            X['orig_center_km'] = X.origin.map(lambda s: geodesic(self.coords[s], self.center).km if s in self.coords else np.nan)
            X['dest_center_km'] = X.destination.map(lambda s: geodesic(self.coords[s], self.center).km if s in self.coords else np.nan)
        return X

class TimeCyclicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['hour_sin'] = np.sin(2 * np.pi * X.hour / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X.hour / 24)
        dow = X['date'].dt.dayofweek
        X['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        X['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        return X

class FeaturePipeline:
    def __init__(self, station_coords: dict):
        self.steps = [
            HolidayTransformer(),
            GeoTransformer(station_coords),
            TimeCyclicTransformer()
        ]
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for step in self.steps:
            X = step.transform(X)
        features = [
            'straight_km', 'orig_center_km', 'dest_center_km',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_holiday'
        ]
        return X[features]

# -------------------- Load model & coords --------------------
@st.cache(allow_output_mutation=True)
def load_model_and_pipeline():
    model = load("final_lightgbm_model.pkl")
    try:
        station_coords = load("station_coords.joblib")
    except Exception:
        df_coor = pd.read_csv("station_coords.csv")
        station_coords = {row['station']:(row['lat'],row['lon']) for _,row in df_coor.iterrows()}
    pipeline = FeaturePipeline(station_coords)
    return model, pipeline, station_coords

# -------------------- Streamlit UI --------------------
st.title("🚆 KTM Komuter Ridership Demo")

# load artifacts
model, pipeline, station_coords = load_model_and_pipeline()
stations = sorted(station_coords.keys())

# input selectors
origin = st.selectbox("Origin station", stations)
destination = st.selectbox("Destination station", stations)
date = st.date_input("Date")
hour = st.slider("Hour of day", 0, 23, 8)

# run prediction button
if st.button("Run prediction"):
    df_raw = pd.DataFrame([{
        "origin": origin,
        "destination": destination,
        "date": pd.to_datetime(date),
        "hour": hour
    }])
    df_feat = pipeline.transform(df_raw)
    pred = model.predict(df_feat)[0]
    df_raw["predicted_ridership"] = pred
    df_ans = df_raw.copy()
    df_ans["is_holiday"] = df_ans['date'].dt.date.apply(lambda d: d in holidays.CountryHoliday("MY"))
    st.table(df_ans)
else:
    st.write("Select parameters and click 'Run prediction' to see results.")

st.markdown("---\n*Built with Streamlit & python-holidays* (national only)")
