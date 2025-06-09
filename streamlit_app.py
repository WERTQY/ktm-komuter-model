import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import holidays

st.set_option('server.showTraceback', True)
st.set_option('global.developmentMode', True)

# Ensure required packages are installed:
# pip install holidays geopy

# 1) Load trained model
@st.cache_resource
def load_model(path: str = 'model.pkl') -> any:
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model('model.pkl')

# 2) Load historical data for feature building (up to cutoff)
@st.cache_data
def load_history(path: str = 'full_history.parquet') -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Convert timestamp to Python date for consistency in comparisons
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

history = load_history()

# 3) Initialize Malaysia holidays
@st.cache_data
def get_my_holidays(start_year: int = 2024, end_year: int = 2030) -> holidays.HolidayBase:
    years = list(range(start_year, end_year + 1))
    return holidays.Malaysia(years=years)

my_holidays = get_my_holidays()

# 4) Static station metadata (coords, states, etc.)
from io import StringIO
from geopy.distance import geodesic

station_coords_csv = """stop_id,stop_name,stop_lat,stop_lon
50000,Sentul,3.183581,101.688802
50300,Batu Kentonmen,3.198485,101.681138
50400,Kampung Batu,3.204892,101.675584
50500,Taman Wahyu,3.21451,101.672178
50600,Batu Caves,3.237796,101.681215
52700,Abdullah Hukum,3.11869,101.67293
52800,Angkasapuri,3.113212,101.673367
52900,Pantai Dalam,3.095607,101.669927
53000,Petaling,3.086485,101.664338
53100,Jalan Templer,3.084013,101.656402
53400,Kampung Dato Harun,3.084828,101.632339
53500,Seri Setia,3.083373,101.61143
53600,Setia Jaya,3.083373,101.61143
53700,Subang Jaya,3.084554,101.587373
53800,Batu Tiga,3.076091,101.559811
54200,Shah Alam,3.056388,101.525302
54400,Padang Jawa,3.052532,101.492742
54500,Bukit Badak,3.036147,101.470176
54700,Klang,3.043078,101.449543
54800,Teluk Pulai,3.04089,101.432153
54900,Teluk Gadong,3.033932,101.424947
55000,Kampung Raja Uda,3.020253,101.41023
55100,Jalan Kastam,3.013128,101.402599
55200,Pelabuhan Klang,2.999323,101.39179
18400,Kepong Sentral,3.208653,101.62849
18600,Kepong,3.202996,101.637381
18700,Segambut,3.186514,101.664032
18800,Putra,3.165399,101.691101
18900,Bank Negara,3.155105,101.693118
19000,Kuala Lumpur,3.139444,101.693333
19100,KL Sentral,3.134167,101.686111
19205,MidValley,3.119211,101.678865
19300,Seputeh,3.113612,101.681474
19400,Salak Selatan,3.098413,101.705055
19600,Bandar Tasek Selatan,3.076229,101.711119
19900,Serdang,3.023404,101.716056
20400,Kajang,2.983222,101.790669
20402,Kajang 2,2.96264,101.79207
20500,UKM,2.939775,101.787623
20900,Bangi,2.904467,101.785943
21300,Batang Benar,2.829904,101.826655
21500,Nilai,2.802356,101.799303
22000,Labu,2.754501,101.826656
22400,Tiroi,2.741459,101.871914
22700,Seremban,2.719169,101.940792
22900,Senawang,2.690138,101.972336
23100,Sungai Gadut,2.660898,101.996158
23900,Rembau,2.593055,102.094653
25100,Pulau Sebang,2.46396,102.226308
15200,Tanjung Malim,3.685142,101.518165
15400,Kalumpang,3.5566,101.5612
16100,Kuala Kubu Bharu,3.553215,101.639591
16300,Rasa,3.500586,101.634113
16500,Batang Kali,3.46838,101.637759
17300,Serendah,3.376172,101.614532
17800,Rawang,3.318955,101.575012
18100,Kuang,3.258267,101.554794
18500,Sungai Buloh,3.206356,101.580128
"""

# Parse station coords and build lookup
df_station_coor = pd.read_csv(StringIO(station_coords_csv))
distance_lookup: dict = {}
coords = df_station_coor.set_index('stop_name')[['stop_lat','stop_lon']].to_dict('index')
for s1, c1 in coords.items():
    for s2, c2 in coords.items():
        distance_lookup[(s1, s2)] = geodesic(
            (c1['stop_lat'], c1['stop_lon']),
            (c2['stop_lat'], c2['stop_lon'])
        ).km

# KTM lines and station metadata
from collections import defaultdict

ktm_lines = [
    {
        "line_id": "Tanjung Malim–Port Klang",
        "stations": [
            {"name": "Tanjung Malim",      "state": "Perak"},
            {"name": "Kuala Kubu Bharu",   "state": "Selangor"},
            {"name": "Rasa",               "state": "Selangor"},
            {"name": "Batang Kali",        "state": "Selangor"},
            {"name": "Serendah",           "state": "Selangor"},
            {"name": "Rawang",             "state": "Selangor"},
            {"name": "Kuang",              "state": "Selangor"},
            {"name": "Sungai Buloh",       "state": "Selangor"},
            {"name": "Kepong Sentral",     "state": "Kuala Lumpur"},
            {"name": "Kepong",             "state": "Kuala Lumpur"},
            {"name": "Segambut",           "state": "Kuala Lumpur"},
            {"name": "Putra",              "state": "Kuala Lumpur"},
            {"name": "Bank Negara",        "state": "Kuala Lumpur"},
            {"name": "Kuala Lumpur",       "state": "Kuala Lumpur"},
            {"name": "KL Sentral",         "state": "Kuala Lumpur"},
            {"name": "Abdullah Hukum",     "state": "Kuala Lumpur"},
            {"name": "Angkasapuri",        "state": "Kuala Lumpur"},
            {"name": "Pantai Dalam",       "state": "Kuala Lumpur"},
            {"name": "Petaling",           "state": "Selangor"},
            {"name": "Jalan Templer",      "state": "Selangor"},
            {"name": "Kampung Dato Harun", "state": "Kuala Lumpur"},
            {"name": "Seri Setia",         "state": "Selangor"},
            {"name": "Setia Jaya",         "state": "Selangor"},
            {"name": "Subang Jaya",        "state": "Selangor"},
            {"name": "Batu Tiga",          "state": "Selangor"},
            {"name": "Shah Alam",          "state": "Selangor"},
            {"name": "Padang Jawa",        "state": "Selangor"},
            {"name": "Bukit Badak",        "state": "Selangor"},
            {"name": "Klang",              "state": "Selangor"},
            {"name": "Teluk Pulai",        "state": "Selangor"},
            {"name": "Teluk Gadong",       "state": "Selangor"},
            {"name": "Kampung Raja Uda",   "state": "Selangor"},
            {"name": "Jalan Kastam",       "state": "Selangor"},
            {"name": "Pelabuhan Klang",    "state": "Selangor"}
        ]
    },
    {
        "line_id": "Batu Caves–Pulau Sebang",
        "stations": [
            {"name": "Batu Caves",              "state": "Selangor"},
            {"name": "Taman Wahyu",            "state": "Kuala Lumpur"},
            {"name": "Kampung Batu",           "state": "Kuala Lumpur"},
            {"name": "Batu Kentonmen",         "state": "Kuala Lumpur"},
            {"name": "Sentul",                 "state": "Kuala Lumpur"},
            {"name": "Putra",                  "state": "Kuala Lumpur"},
            {"name": "Bank Negara",            "state": "Kuala Lumpur"},
            {"name": "Kuala Lumpur",           "state": "Kuala Lumpur"},
            {"name": "KL Sentral",             "state": "Kuala Lumpur"},
            {"name": "MidValley",              "state": "Kuala Lumpur"},
            {"name": "Seputeh",                "state": "Kuala Lumpur"},
            {"name": "Salak Selatan",          "state": "Kuala Lumpur"},
            {"name": "Bandar Tasek Selatan",   "state": "Kuala Lumpur"},
            {"name": "Serdang",                "state": "Selangor"},
            {"name": "Kajang",                 "state": "Selangor"},
            {"name": "Kajang 2",               "state": "Selangor"},
            {"name": "UKM",                    "state": "Selangor"},
            {"name": "Bangi",                  "state": "Selangor"},
            {"name": "Batang Benar",           "state": "Negeri Sembilan"},
            {"name": "Nilai",                  "state": "Negeri Sembilan"},
            {"name": "Labu",                   "state": "Negeri Sembilan"},
            {"name": "Tiroi",                  "state": "Negeri Sembilan"},
            {"name": "Seremban",               "state": "Negeri Sembilan"},
            {"name": "Senawang",               "state": "Negeri Sembilan"},
            {"name": "Sungai Gadut",           "state": "Negeri Sembilan"},
            {"name": "Rembau",                 "state": "Negeri Sembilan"},
            {"name": "Pulau Sebang",           "state": "Malacca"}
        ]
    }
]
# Build station->lines and station->state lookup
station2lines: dict = defaultdict(list)
station2state: dict = {}
for line in ktm_lines:
    lid = line['line_id']
    for station in line['stations']:
        station2lines[station['name']].append(lid)
        station2state[station['name']] = station['state']

# Interchange stations (correct spelling)
station_connections = {
    "Bandar Tasik Selatan": {
        "connections": ["LRT_Sri_Petaling_Line", "ERL_KLIA_Transit"],
        "interchange_type": "direct"
    },
    "KL Sentral": {
        "connections": ["LRT_Kelana_Jaya_Line", "MRT_Kajang_Line", "Monorail",
                        "ERL_KLIA_Transit", "ERL_KLIA_Ekspres", "GoKL_Bus"],
        "interchange_type": "direct"
    },
    "Kuala Lumpur": {
        "connections": ["LRT_Ampang_Line", "LRT_Sri_Petaling_Line", "GoKL_Bus"],
        "interchange_type": "indirect"
    },
    "Bank Negara": {
        "connections": ["LRT_Ampang_Line", "LRT_Sri_Petaling_Line", "GoKL_Bus"],
        "interchange_type": "indirect"
    },
    "Putra": {
        "connections": ["LRT_Ampang_Line", "LRT_Sri_Petaling_Line", "GoKL_Bus"],
        "interchange_type": "indirect"
    },
    "Subang Jaya": {
        "connections": ["LRT_Kelana_Jaya_Line"],
        "interchange_type": "direct"
    },
    "Abdullah Hukum": {
        "connections": ["LRT_Kelana_Jaya_Line"],
        "interchange_type": "direct"
    },
    "Kajang": {
        "connections": ["MRT_Kajang_Line", "MRT_Feeder_Bus"],
        "interchange_type": "direct"
    },
    "Sungai Buloh": {
        "connections": ["MRT_Putrajaya_Line"],
        "interchange_type": "direct"
    },
    "Setia Jaya": {
        "connections": ["BRT_Sunway_Line"],
        "interchange_type": "direct"
    },
    "Serdang": {
        "connections": ["MRT_Feeder_Bus"],
        "interchange_type": "feeder"
    },
    "Kepong Sentral": {
        "connections": ["MRT_Putrajaya_Line"],
        "interchange_type": "direct"
    },
    "Kampung Batu": {
        "connections": ["MRT_Putrajaya_Line"],
        "interchange_type": "direct"
    }
}

central_corridor_stations = ['Putra','Bank Negara','Kuala Lumpur','KL Sentral']

# 5) Feature builder
def build_features(df_hist: pd.DataFrame, origin: str, dest: str, date: datetime.date, hour: int) -> pd.DataFrame:
    row = {
        'origin': origin,
        'destination': dest,
        'date': date,
        'hour': hour,
    }
    # Temporal features
    dow = date.weekday()
    row['day_of_week'] = dow
    row['is_holiday'] = int(date in my_holidays)
    # Cyclical features
    row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    row['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    row['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    # Historical lags
    odh = df_hist[
        (df_hist.origin == origin) &
        (df_hist.destination == dest) &
        (df_hist.hour == hour)
    ]
    prev1 = odh[odh.date == (date - datetime.timedelta(days=1))]['ridership']
    row['lag1_ridership'] = float(prev1.values[0]) if len(prev1) > 0 else 0.0
    window = odh[ (odh.date >= (date - datetime.timedelta(days=7))) & (odh.date < date) ]['ridership']
    row['roll7_mean_od'] = float(window.mean()) if len(window) > 0 else 0.0
    row['weekly_pattern_std'] = float(odh['ridership'].std()) if len(odh) > 1 else 0.0
    # Spatial features
    dist = distance_lookup.get((origin, dest), np.nan)
    row['straight_line_distance_km'] = dist
    row['trip_duration_estimate_min'] = (dist / 120) * 60 if not np.isnan(dist) else np.nan
    row['same_state'] = int(station2state.get(origin) == station2state.get(dest))
    row['is_origin_central'] = int(origin in central_corridor_stations)
    row['is_dest_central'] = int(dest in central_corridor_stations)
    row['origin_is_interch'] = int(origin in station_connections)
    row['dest_is_interch'] = int(dest in station_connections)
    return pd.DataFrame([row])

# 6) UI
st.title('KTM Komuter Ridership Forecast')
st.markdown('Select origin, destination, date (>= 2025-05-13) and hour to forecast ridership.')

origins = history.origin.unique()
destinations = history.destination.unique()
origin = st.selectbox('Origin', sorted(origins))
destination = st.selectbox('Destination', sorted(destinations))

selected_date = st.date_input('Date', min_value=datetime.date(2025, 5, 13), value=datetime.date(2025, 5, 13))
hour = st.slider('Hour', 0, 23, 8)

# Recursive forecasting
def recursive_forecast():
    start_date = history.date.max() + datetime.timedelta(days=1)
    end_date = selected_date
    if end_date < start_date:
        st.warning(f"Please select a date on or after {start_date.isoformat()}.")
        return
    dates = pd.date_range(start_date, end_date, freq='D').date
    preds = []
    hist = history.copy()
    for current_date in dates:
        feats = build_features(hist, origin, destination, current_date, hour)
        X = feats[model.feature_names_in_]
        yhat = model.predict(X)[0]
        preds.append({'date': current_date, 'predicted_ridership': float(yhat)})
        new = feats.copy()
        new['ridership'] = yhat
        hist = pd.concat([hist, new], ignore_index=True)
    result = pd.DataFrame(preds)
    st.line_chart(result.set_index('date')['predicted_ridership'])
    st.dataframe(result)

# Single-step prediction
def single_predict():
    feats = build_features(history, origin, destination, selected_date, hour)
    X = feats[model.feature_names_in_]
    yhat = model.predict(X)[0]
    st.write(f"Predicted ridership on {selected_date} at {hour}:00 → {yhat:.1f} riders")

if st.button('Predict Series'):
    recursive_forecast()
elif st.button('Predict Single'):
    single_predict()

# To run: streamlit run streamlit_app.py
