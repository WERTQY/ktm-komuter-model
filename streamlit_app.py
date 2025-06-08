import streamlit
import pandas as pd
import numpy as np
import pickle
import datetime
from collections import defaultdict
from io import StringIO
from geopy.distance import geodesic
import holidays
import category_encoders as ce

# ────────────────────────────────────────────────────────────────
# 1) Load model, encoder, feature list, and historical data
# ────────────────────────────────────────────────────────────────
@streamlit.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl","rb"))
    enc_hi = pickle.load(open("enc_hi.pkl","rb"))
    feat_cols = pickle.load(open("feature_columns.pkl","rb"))
    # historical ridership up to 2025-05-12
    hist = pd.read_parquet("full_history.parquet", columns=[
        "date","hour","origin","destination","ridership"
    ])
    return model, enc_hi, feat_cols, hist

model, enc_hi, feature_columns, hist_df = load_artifacts()

streamlit.title("KTM Komuter Recursive Ridership Predictor")

# ────────────────────────────────────────────────────────────────
# 2) Precompute static maps (lines, states, coords, distances, holidays…)
# ────────────────────────────────────────────────────────────────
# 2.1 ktm_lines (fill in your full list)
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

station2lines = defaultdict(list)
station2state = {}
for line in ktm_lines:
    lid = line["line_id"]
    for s in line["stations"]:
        name = s["name"].strip()
        station2lines[name].append(lid)
        station2state[name] = s["state"]
station_list = sorted(station2state)

# 2.2 station coords CSV (fill in all rows)
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
df_coords = pd.read_csv(StringIO(station_coords_csv)).drop(columns=["stop_id"])
df_coords.rename(columns={"stop_name":"station","stop_lat":"lat","stop_lon":"lon"}, inplace=True)
coord_map = df_coords.set_index("station")[["lat","lon"]].to_dict("index")

# 2.3 Distances to KL Sentral
center = (coord_map["KL Sentral"]["lat"], coord_map["KL Sentral"]["lon"])
station_to_center = {
    st: geodesic((c["lat"],c["lon"]), center).km
    for st,c in coord_map.items()
}

# 2.4 Nearest interchange distance
interchange_stations = [
    "KL Sentral","Bandar Tasek Selatan","Kuala Lumpur",
    "Bank Negara","Putra","Subang Jaya","Kajang"
]
inter_coords = [(coord_map[s]["lat"],coord_map[s]["lon"]) for s in interchange_stations]
station_to_interchange = {}
for station,c in coord_map.items():
    if station in interchange_stations:
        station_to_interchange[station] = 0.0
    else:
        station_to_interchange[station] = min(
            geodesic((c["lat"],c["lon"]), ic).km for ic in inter_coords
        )

# 2.5 Station density within 5 km
def density(st):
    lat0,lon0 = coord_map[st]["lat"], coord_map[st]["lon"]
    return sum(
        1 for o,c in coord_map.items()
        if o!=st and geodesic((lat0,lon0),(c["lat"],c["lon"])).km<=5
    )
station_density = {st:density(st) for st in coord_map}

# 2.6 Holiday calendars
# nat_hols = holidays.Malaysia()
# state_hols = {st: holidays.Malaysia(subdiv=st) for st in set(station2state.values())}
# Holiday calendars: try state‐level, else fallback to national
nat_hols = holidays.Malaysia()
state_hols = {}
for state in set(station2state.values()):
    try:
        state_hols[state] = holidays.Malaysia(subdiv=state)
    except NotImplementedError:
        # fallback for unsupported subdivision names
        state_hols[state] = nat_hols
festivals = {"Chinese New Year","Thaipusam","Hari Raya Puasa","Hari Raya Haji","Deepavali"}

# 2.7 Precompute per-origin/hour average ridership from hist_df
avg_map = (
    hist_df
    .groupby(["origin","hour"])["ridership"]
    .mean()
    .to_dict()
)

# ────────────────────────────────────────────────────────────────
# 3) Helper: build all non-history features for a single timestamp
# ────────────────────────────────────────────────────────────────
def prepare_static_feats(df):
    """
    Takes a DataFrame with columns:
      ['origin','destination','date','hour']
    and returns same DataFrame with all static features computed:
      lines, states, is_peak, distances, cyclic, holiday, interactions…
    """
    df = df.copy()
    # date parts
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["day_of_mon"]   = df["date"].dt.day
    df["quarter"]      = df["date"].dt.quarter
    df["week_of_yr"]   = df["date"].dt.isocalendar().week.astype(int)
    # line
    def pick_line(o,d):
        ol,dl = station2lines[o], station2lines[d]
        c = set(ol)&set(dl)
        if   len(c)==1: return c.pop()
        elif len(c)>1:  return "Central Corridor"
        else:           return f"{ol[0]} -> {dl[0]} (Transfer)"
    df["line"] = df.apply(lambda r: pick_line(r.origin,r.destination), axis=1)
    # states & interch
    df["origin_state"]      = df["origin"].map(station2state)
    df["destination_state"] = df["destination"].map(station2state)
    df["is_origin_central"] = df["origin"].isin(interchange_stations).astype("uint8")
    df["is_dest_central"]   = df["destination"].isin(interchange_stations).astype("uint8")
    df["origin_is_interch"]= df["origin"].isin(interchange_stations).astype("uint8")
    df["dest_is_interch"]  = df["destination"].isin(interchange_stations).astype("uint8")
    # connections (zero if you don’t have station_connections)
    transports = [
      "LRT_Sri_Petaling_Line","LRT_Kelana_Jaya_Line","MRT_Kajang_Line",
      "Monorail","ERL_KLIA_Transit","ERL_KLIA_Ekspres",
      "BRT_Sunway_Line","GoKL_Bus","MRT_Feeder_Bus","MRT_Putrajaya_Line"
    ]
    for t in transports:
        df[f"orig_conn_{t}"] = 0
        df[f"dest_conn_{t}"] = 0
    # distances
    df["straight_line_distance_km"] = df.apply(
        lambda r: geodesic(
            (coord_map[r.origin]["lat"],coord_map[r.origin]["lon"]),
            (coord_map[r.destination]["lat"],coord_map[r.destination]["lon"])
        ).km, axis=1
    )
    df["dist_orig_center"]    = df["origin"].map(station_to_center)
    df["dist_dest_center"]    = df["destination"].map(station_to_center)
    df["orig_dist_to_interchange"] = df["origin"].map(station_to_interchange)
    df["dest_dist_to_interchange"] = df["destination"].map(station_to_interchange)
    df["orig_station_density"]     = df["origin"].map(station_density)
    df["dest_station_density"]     = df["destination"].map(station_density)
    # cyclic
    df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]    = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]    = np.cos(2*np.pi*df["day_of_week"]/7)
    df["month_sin"]  = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"]  = np.cos(2*np.pi*(df["month"]-1)/12)
    # holidays
    df["orig_holiday"] = df.apply(
        lambda r: r["date"] in state_hols.get(r["origin_state"],nat_hols), axis=1
    ).astype("uint8")
    df["dest_holiday"] = df.apply(
        lambda r: r["date"] in state_hols.get(r["destination_state"],nat_hols), axis=1
    ).astype("uint8")
    df["is_holiday"] = (df["orig_holiday"]|df["dest_holiday"]).astype("uint8")
    # near-festival
    fest_dates = [
        pd.Timestamp(d)   # convert date → Timestamp
        for d,name in nat_hols.items() 
        if name in festivals
    ]
    df["near_big_holiday"] = df["date"].apply(
        lambda d: any(abs((d-f).days)<=1 for f in fest_dates)
    ).astype("uint8")
    # interactions
    df["orig_dest_pair"]  = df["origin"]+"_"+df["destination"]
    df["is_peak"]         = ((df["hour"].between(7,9)|df["hour"].between(17,19))).astype("uint8")
    df["line_peak_combo"] = df["line"] + "_" + df["is_peak"].astype(str)
    df["same_state"]      = (df["origin_state"]==df["destination_state"]).astype("uint8")
    df["both_interch"]    = (df["origin_is_interch"]&df["dest_is_interch"]).astype("uint8")
    df["same_corridor"]   = (df["is_origin_central"]&df["is_dest_central"]).astype("uint8")
    df["hour_weekend"]    = df["hour"]*((df["day_of_week"]>=5).astype(int))
    df["peak_on_corridor"]= (df["is_peak"]&df["same_corridor"]).astype("uint8")
    return df

# ────────────────────────────────────────────────────────────────
# 4) User Inputs
# ────────────────────────────────────────────────────────────────
origin      = streamlit.selectbox("Origin", station_list)
destination = streamlit.selectbox("Destination", station_list)
travel_date = streamlit.date_input("Predict Through Date", min_value=datetime.date(2025, 5, 13))
hour        = streamlit.number_input("Hour of Day", min_value=0, max_value=23, value=0)

if streamlit.button("Run Recursive Forecast"):
    # 5) Seed series with real ridership up to 2025-05-12 at this hour:
    hist = hist_df[
        (hist_df.origin==origin)&
        (hist_df.destination==destination)&
        (hist_df.hour==hour)
    ].copy()
    hist.index = pd.to_datetime(hist.date) + pd.to_timedelta(hist.hour, "h")
    hist = hist.sort_index()["ridership"]

    # 6) Build prediction timestamps (each day at chosen hour)
    start_ts = pd.Timestamp("2025-05-13") + pd.Timedelta(hours=hour)
    end_ts   = pd.Timestamp.combine(travel_date, datetime.time(hour))
    pred_times = pd.date_range(start=start_ts, end=end_ts, freq="24H")

    # … after computing pred_times …

# 2.7 Extract the three target‐encode mappings once up front
# 2.7 Extract the three target‐encode mappings once up front
    mp = enc_hi.mapping
    if isinstance(mp, dict):
        # mapping was saved as {col: pd.Series(mapping), ...}
        origin_map = mp['origin']
        dest_map   = mp['destination']
        pair_map   = mp['orig_dest_pair']
    else:
        # fallback for list-of-dicts style
        def _get_map_list(mapping, col):
            for m in mapping:
                if m.get('col') == col:
                    return m.get('mapping')
            raise KeyError(f"No mapping found for {col}")
        origin_map = _get_map_list(mp, 'origin')
        dest_map   = _get_map_list(mp, 'destination')
        pair_map   = _get_map_list(mp, 'orig_dest_pair')
    
    global_mean = enc_hi._global_mean


    # B) build one-row df
    row = pd.DataFrame([{
        "origin": origin,
        "destination": destination,
        "orig_dest_pair": f"{origin}_{destination}",
        "date": ts.normalize(),
        "hour": hour,
        "lag1_ridership": lag1,
        "lag24_ridership": lag24,
        "delta_vs_yday": delta,
        "roll7_mean_od": roll7,
        "weekly_pattern_std": std7,
        "avg_ridership_origin_hour": avg_oh
    }])

    # C) static
    feat = prepare_static_feats(row)

    # D) manual target‐encode
    feat['origin_te']         = feat['origin'].map(origin_map).fillna(global_mean)
    feat['destination_te']    = feat['destination'].map(dest_map).fillna(global_mean)
    feat['orig_dest_pair_te'] = feat['orig_dest_pair'].map(pair_map).fillna(global_mean)

    # E) drop raw ID cols
    feat.drop(columns=['origin','destination','orig_dest_pair'], inplace=True, errors='ignore')

    # F) one-hot low-card
    low_card = ['line','origin_state','destination_state',
                'origin_interch_type','dest_interch_type',
                'line_peak_combo']
    feat = pd.get_dummies(feat, columns=low_card, drop_first=True, dtype=np.uint8)

    # G) align
    enc = feat.reindex(columns=feature_columns, fill_value=0)

    # H) predict & append
    yhat = model.predict(enc)[0]
    series.at[ts] = yhat
    results.append({"date": ts.date(), "predicted_ridership": int(yhat)})

# 8) Show results
df_res = pd.DataFrame(results)
st.subheader("Recursive Forecast Results")
st.dataframe(df_res, use_container_width=True)
