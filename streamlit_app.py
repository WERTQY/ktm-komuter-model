import streamlit as st
import pandas as pd
import numpy as np
import holidays
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load

# -------------------- Full Preprocessing Pipeline --------------------
class FullFeaturePipeline:
    def __init__(self, station_coords: dict, holiday_df: pd.DataFrame):
        self.station_coords = station_coords
        # holiday_df: columns ['date','states'] mapping date→states list
        # convert to dict date_str→set(states)
        self.ph = {
            row['date'].strftime('%Y-%m-%d'): set(s.split(', '))
            for _, row in holiday_df.iterrows()
        }
        # define station→state from coords dict or separate mapping
        # here assume station_coords keys are names and coords includes state
        # you could load a station2state mapping similarly
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ensure types
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['hour'].astype(int)

        # line assignment and station state mapping
        # define station2lines & station2state as in preprocessing code
        station2lines, station2state = self._load_line_state_maps()
        df['line'] = df.apply(lambda r: self._pick_line(r.origin, r.destination, station2lines), axis=1)
        df['origin_state'] = df.origin.map(station2state)
        df['destination_state'] = df.destination.map(station2state)
        # holiday flags
        df['orig_holiday'] = df.apply(lambda r: r.origin_state in self.ph.get(r.date.strftime('%Y-%m-%d'), set()), axis=1)
        df['dest_holiday'] = df.apply(lambda r: r.destination_state in self.ph.get(r.date.strftime('%Y-%m-%d'), set()), axis=1)
        df['is_holiday'] = df[['orig_holiday','dest_holiday']].any(axis=1).astype(int)

        # geospatial features
        df[['origin_lat','origin_lon']] = df.origin.map(self.station_coords).apply(pd.Series)
        df[['dest_lat','dest_lon']]      = df.destination.map(self.station_coords).apply(pd.Series)
        df['straight_km'] = df.apply(lambda r: geodesic((r.origin_lat,r.origin_lon),(r.dest_lat,r.dest_lon)).km, axis=1)
        center = self.station_coords['KL Sentral']
        df['orig_center_km'] = df.origin.map(lambda s: geodesic(self.station_coords[s], center).km)
        df['dest_center_km'] = df.destination.map(lambda s: geodesic(self.station_coords[s], center).km)

        # cyclical time
        df['hour_sin'] = np.sin(2*np.pi*df.hour/24)
        df['hour_cos'] = np.cos(2*np.pi*df.hour/24)
        dow = df.date.dt.dayofweek
        df['dow_sin'] = np.sin(2*np.pi*dow/7)
        df['dow_cos'] = np.cos(2*np.pi*dow/7)

        # lag features
        df = df.sort_values(['origin','destination','date','hour']).reset_index(drop=True)
        grp = df.groupby(['origin','destination'])['ridership']
        df['lag1'] = grp.shift(1).fillna(0)
        df['lag24'] = grp.shift(24).fillna(0)
        df['roll7'] = (grp.shift(1).rolling(7, min_periods=1).mean()).fillna(0)

        features = [
            'straight_km','orig_center_km','dest_center_km',
            'hour_sin','hour_cos','dow_sin','dow_cos',
            'is_holiday','lag1','lag24','roll7'
        ]
        return df[features]

    def _load_line_state_maps(self):
        # replicate station2lines and station2state definitions
        # ... (use the code from preprocessing) ...
        return station2lines, station2state

    def _pick_line(self, orig, dest, station2lines):
        ol = station2lines.get(orig, []); dl = station2lines.get(dest, [])
        common = set(ol)&set(dl)
        if len(common)==1: return common.pop()
        if len(common)>1: return 'Central'
        if ol and dl: return f"{ol[0]}->{dl[0]}"
        return np.nan

# -------------------- Load artifacts --------------------
@st.cache(allow_output_mutation=True)
def load_artifacts():
    model = load('model.joblib')
    station_coords = load('station_coords.joblib')  # {name:(lat,lon)}
    holiday_df = pd.read_parquet('df_public_holiday_combined.parquet')
    hist = pd.read_parquet('df_ridership_clean.parquet')
    hist = hist[hist['date']>=pd.to_datetime('2024-01-01')]
    return model, station_coords, holiday_df, hist

# -------------------- Streamlit UI --------------------
st.title('🚆 KTMB Komuter Ridership Forecast')
model, station_coords, holiday_df, hist_df = load_artifacts()

stations = sorted(station_coords.keys())
origin = st.selectbox('Origin', stations)
destination = st.selectbox('Destination', stations)
end_date = st.date_input('Forecast until date', value=hist_df['date'].max().date())
hour = st.slider('Hour', 0,23,8)

# compute base last timestamp
g_last = hist_df.copy()
last_date = g_last['date'].max(); last_hour = g_last[g_last['date']==last_date]['hour'].max()
last_ts = last_date + pd.Timedelta(hours=int(last_hour))
if end_date <= last_ts.date(): st.warning('Pick after last data'); st.stop()

# iterative forecast
pipe = FullFeaturePipeline(station_coords, holiday_df)
df = hist_df.copy()
current = last_ts + pd.Timedelta(days=1)
results=[]; times=[]
while current.date()<=pd.to_datetime(end_date).date():
    df = pd.concat([df, pd.DataFrame([{
        'origin':origin,'destination':destination,
        'date':current.normalize(),'hour':current.hour,'ridership':np.nan
    }])], ignore_index=True)
    X = pipe.transform(df)
    pred = model.predict(X.tail(1))[0]
    results.append(pred); times.append(current)
    df.at[df.index[-1],'ridership']=pred
    current+=pd.Timedelta(days=1)

out = pd.DataFrame({'timestamp':times,'prediction':results}).set_index('timestamp')
st.line_chart(out)
