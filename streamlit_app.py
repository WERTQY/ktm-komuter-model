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
    model = load("final_lightgbm_model.joblib")
    try:
        station_coords = load("station_coords.joblib")
    except Exception:
        df_coor = pd.read_csv("station_coords.csv")
        station_coords = {row['station']:(row['lat'],row['lon']) for _,row in df_coor.iterrows()}
    pipeline = FeaturePipeline(station_coords)
    return model, pipeline, station_coords

# -------------------- Load historical data --------------------
@st.cache(allow_output_mutation=True)
def load_history():
    # expect a Parquet or CSV historical file with columns: origin, destination, date, hour, ridership
    try:
        hist = pd.read_parquet("full_history.parquet")
    except Exception:
        hist = pd.read_csv("full_history.csv", parse_dates=["date"])
    # filter to 2024 onwards
    hist = hist[hist['date'] >= pd.to_datetime("2024-01-01")]
    # ensure correct types
    hist['date'] = pd.to_datetime(hist['date'])
    hist['hour'] = hist['hour'].astype(int)
    return hist

# -------------------- Streamlit UI --------------------
st.title("🚆 KTM Komuter Ridership Multi-Step Demo")

# load artifacts
model, pipeline, station_coords = load_model_and_pipeline()
stations = sorted(station_coords.keys())
hist_df = load_history()

# input selectors
origin = st.selectbox("Origin station", stations)
destination = st.selectbox("Destination station", stations)
target_date = st.date_input("Forecast until date", value=hist_df['date'].max().date())
hour = st.slider("Hour of day for forecasts", 0, 23, 8)

# determine last available datetime in history (most recent date and hour)
last_hist_date = hist_df['date'].max()
last_hist_hour = hist_df.loc[hist_df['date']==last_hist_date, 'hour'].max()
last_dt = pd.to_datetime(last_hist_date) + pd.to_timedelta(last_hist_hour, unit='h')

if target_date <= last_dt.date():
    st.warning(f"Please choose a date after your last data date {last_dt.date()}")
else:
    if st.button("Run multi-step forecast"):
        # iterative forecasting
        results = []
        timestamps = []
        df = hist_df.copy()
        current = last_dt + pd.Timedelta(days=1)
        end_date = pd.to_datetime(target_date)
        while current.date() <= end_date.date():
            new_row = {
                "origin": origin,
                "destination": destination,
                "date": current.normalize(),
                "hour": current.hour,
                "ridership": np.nan
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            X_all = pipeline.transform(df.assign(date=pd.to_datetime(df['date'])))
            X_new = X_all.tail(1)
            pred = model.predict(X_new)[0]
            results.append(float(pred))
            timestamps.append(current)
            df.at[df.index[-1], 'ridership'] = pred
            current += pd.Timedelta(days=1)
        out = pd.DataFrame({"timestamp": timestamps, "predicted_ridership": results})
        out.set_index('timestamp', inplace=True)
        st.line_chart(out)
    else:
        st.write("Select inputs and click 'Run multi-step forecast' to see predictions from last data up to your date.")

st.markdown("---
*Forecasts start from 2024 and use full history up to last available date.*")
