import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
from collections import defaultdict
import holidays
from io import StringIO

# --- 1) Static lookups ---
STATION_COORD_CSV = """stop_id,stop_name,stop_lat,stop_lon
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

coord_df = pd.read_csv(StringIO(STATION_COORD_CSV)).drop(columns=['stop_id'])
coord_df.rename(columns={'stop_name':'station','stop_lat':'lat','stop_lon':'lon'}, inplace=True)
coord_df.set_index('station', inplace=True)
station_to_coord = coord_df.to_dict('index')

# KTM lines & states
ktm_lines = [
    # full ktm_lines definitions omitted for brevity
]
station2lines = defaultdict(list)
station2state = {}
for line in ktm_lines:
    lid = line['line_id']
    for st in line['stations']:
        name=st['name']; station2lines[name].append(lid)
        station2state[name]=st['state']

# connections
station_connections = {
    # full connections dict
}
all_transports = [
    # list of transport modes
]
central_corr = {'Putra','Bank Negara','Kuala Lumpur','KL Sentral'}

# holiday calendar
mal_holidays = holidays.Malaysia()

# --- 2) Feature Builder ---
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        df = X.copy()
        # core types
        df['date']=pd.to_datetime(df['date']); df['hour']=df['hour'].astype(int)
        # time diffs
        df['timestamp']=df['date']+pd.to_timedelta(df['hour'],unit='h')
        df.sort_values(['origin','destination','timestamp'],inplace=True)
        df['time_diff_hours']=df.groupby(['origin','destination'])['timestamp'].diff().dt.total_seconds().fillna(0)/3600
        df['is_first_record'] = df['time_diff_hours']==0
        # date features
        df['day_of_week']=df['date'].dt.dayofweek
        df['year']=df['date'].dt.year
        df['month']=df['date'].dt.month
        df['day_of_mon']=df['date'].dt.day
        df['quarter']=df['date'].dt.quarter
        df['week_of_yr']=df['date'].dt.isocalendar().week.astype(int)
        # holiday flags
        df['orig_holiday']=df['origin_state'].map(lambda st: df['date'].dt.date.isin(mal_holidays))
        df['dest_holiday']=df['destination_state'].map(lambda st: df['date'].dt.date.isin(mal_holidays))
        df['any_holiday']=df['orig_holiday']|df['dest_holiday']
        df['is_holiday']=df['any_holiday']
        # cyclic
        df['hour_sin']=np.sin(2*np.pi*df['hour']/24)
        df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
        df['dow_sin']=np.sin(2*np.pi*df['day_of_week']/7)
        df['dow_cos']=np.cos(2*np.pi*df['day_of_week']/7)
        df['month_sin']=np.sin(2*np.pi*(df['month']-1)/12)
        df['month_cos']=np.cos(2*np.pi*(df['month']-1)/12)
        # line & states
        df['line']=df.apply(lambda r: pick_line(r.origin,r.destination),axis=1)
        df['origin_state']=df['origin'].map(station2state)
        df['destination_state']=df['destination'].map(station2state)
        # orig_dest pair
        df['orig_dest_pair']=df['origin']+'_'+df['destination']
        # spatial coords
        for side in ['origin','destination']:
            df[[f'{side}_lat',f'{side}_lon']]=coord_df.loc[df[side]].values
        # distances
        df['straight_line_distance_km']=df.apply(lambda r: geodesic((r.origin_lat,r.origin_lon),(r.destination_lat,r.destination_lon)).km,axis=1)
        df['trip_duration_estimate_min']=df['straight_line_distance_km']/120*60
        # central & interchange
        df['is_origin_central']=df['origin'].isin(central_corr)
        df['is_dest_central']=df['destination'].isin(central_corr)
        df['origin_is_interch']=df['origin'].isin(station_connections)
        df['dest_is_interch']=df['destination'].isin(station_connections)
        df['origin_interch_type']=df['origin'].map(lambda s: station_connections.get(s,{}).get('interchange_type','none'))
        df['dest_interch_type']=df['destination'].map(lambda s: station_connections.get(s,{}).get('interchange_type','none'))
        for t in all_transports:
            df[f'orig_conn_{t}']=df['origin'].map(lambda s: t in station_connections.get(s,{}).get('connections',[]))
            df[f'dest_conn_{t}']=df['destination'].map(lambda s: t in station_connections.get(s,{}).get('connections',[]))
        # interaction flags
        df['same_state']=(df['origin_state']==df['destination_state']).astype('uint8')
        df['both_interch']=(df['origin_is_interch']&df['dest_is_interch']).astype('uint8')
        df['same_corridor']=(df['is_origin_central']&df['is_dest_central']).astype('uint8')
        df['line_peak_combo']=df['line']+'_'+df['is_peak'].astype(str)
        # freq encoding
        freq=df['orig_dest_pair'].value_counts(normalize=True)
        df['pair_freq']=df['orig_dest_pair'].map(freq).astype('float32')
        # holiday proximity
        hol_dates=sorted(df.loc[df['is_holiday'],'date'].dt.normalize().unique())
        df['next_hol']=pd.merge_asof(df[['date']].sort_values('date'),pd.DataFrame({'hol':hol_dates}),left_on='date',right_on='hol',direction='forward')['hol']
        df['days_to_next_holiday']=(df['next_hol']-df['date']).dt.days.fillna(999).astype('int16')
        bins=[-999,-3,-1,0,1,3,999]; labels=['>3_before','1-3_before','eve','holiday','1_after','2-3_after']
        df['holiday_window']=pd.cut(df['days_to_next_holiday'],bins=bins,labels=labels).cat.codes.astype('int8')
        # rolling & lag features
        daily=df.groupby(['origin','destination','hour','date'])['ridership'].sum().unstack(fill_value=0)
        roll7=daily.shift(axis=1).rolling(7,axis=1,min_periods=1).mean().stack().rename('roll7_mean_od').reset_index()
        std=daily.std(axis=1,ddof=0).rename('weekly_pattern_std').reset_index()
        df=df.merge(roll7,on=['origin','destination','hour','date'],how='left').merge(std,on=['origin','destination','hour'],how='left')
        df['roll7_mean_od'].fillna(0,inplace=True); df['weekly_pattern_std'].fillna(0,inplace=True)
        # origin-hour average
        n_days=daily.shape[1]; avg=daily.sum(level=[0,2]).div(n_days).rename('avg_ridership_origin_hour').reset_index()
        df=df.merge(avg,on=['origin','hour'],how='left')
        # lags
        grp=df.groupby(['origin','destination'])['ridership']
        df['lag1_ridership']=grp.shift(1).fillna(0).astype('float32')
        df['lag24_ridership']=grp.shift(24).fillna(0).astype('float32')
        df['delta_vs_yday']=(df['ridership']-df['lag24_ridership']).astype('float32')
        # drop raw columns
        drop_cols=['hour','timestamp','date','origin_lat','origin_lon','dest_lat','dest_lon','trip_duration_estimate_min','next_hol']
        df.drop(columns=drop_cols,errors='ignore',inplace=True)
        return df

# --- 3) Preprocessor & Model ---
HI_CARD_COLS=['origin','destination','orig_dest_pair']
LOW_CARD_COLS=['line','origin_state','destination_state','origin_interch_type','dest_interch_type','line_peak_combo']
# booleans auto-captured

numeric_feats=[
 'straight_line_distance_km','year','day_of_mon','quarter','week_of_yr',
 'time_diff_hours','hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos',
 'pair_freq','days_to_next_holiday','roll7_mean_od','weekly_pattern_std',
 'avg_ridership_origin_hour','lag1_ridership','lag24_ridership','delta_vs_yday',
 'hour_weekend','dist_orig_center','dist_dest_center','orig_dist_to_interchange','dest_dist_to_interchange',
 'orig_station_density','dest_station_density','bearing_sin','bearing_cos'
]

preprocessor=ColumnTransformer([
 ('num','passthrough',numeric_feats),
 ('ohe',OneHotEncoder(handle_unknown='ignore'),LOW_CARD_COLS),
 ('tgt',TargetEncoder(cols=HI_CARD_COLS,min_samples_leaf=50,smoothing=10),HI_CARD_COLS)
],remainder='passthrough')

model=RandomForestRegressor(n_estimators=200,random_state=42)
pipeline=Pipeline([('features',FeatureBuilder()),('prep',preprocessor),('model',model)])

# --- 4) Train & Save ---
if __name__=='__main__':
 df=pd.read_parquet('df_ridership_clean.parquet')
 X=df[['origin','destination','date','hour']]; y=df['ridership']
 pipeline.fit(X,y)
 with open('pipeline.pkl','wb') as f: pickle.dump(pipeline,f)
 print('✅ pipeline.pkl saved.')
