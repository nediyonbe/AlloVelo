# import modules
import pandas as pd
import glob
import numpy as np
import datetime
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats
from datetime import timedelta
from sklearn.metrics import r2_score
###############################################################################
# Import stations and their coordinates
###############################################################################
path_stations = 'C:/Users/Ali/Documents/Insight/Bixi/Stations' 
all_files_stations = glob.glob(path_stations + "/*.csv")
listy_stations = []
counter_f_stations = 0
for f in all_files_stations:
    col_names_stations = ['station_code', 'station_name', 'station_latitude', 
                          'station_longitude', 'station_year']
    df = pd.read_csv(f, skiprows=1, names=col_names_stations)
    listy_stations.append(df)
df_stations = pd.concat(listy_stations, axis=0, ignore_index=True)
df_stations.head()

# Pickup and drops will be regrouped by clusters
# allocate stations on a 5x5 matrix limited by the following coordinates:
# N: 45.56, S:45.42, W:-73.68, E:-73.49
lat_max = df_stations['station_latitude'].max()
lat_min = df_stations['station_latitude'].min()
lon_max = df_stations['station_longitude'].max()
lon_min = df_stations['station_longitude'].min()
num_clusters = 5  # this creates a grid of 25 zones provided that there is a statition in that zone
lat_interval = (lat_max - lat_min) / num_clusters
lon_interval = (lon_max - lon_min) / num_clusters

for clustorizer_lat in range(1, num_clusters+1):
    print(clustorizer_lat)
    for clustorizer_lon in range(1, num_clusters+1):
            df_stations.loc[(df_stations['station_latitude'] >= lat_min + lat_interval * (clustorizer_lat - 1)) &
                            (df_stations['station_latitude'] <= lat_min + lat_interval * clustorizer_lat) &
                            (df_stations['station_longitude'] >= lon_min + lon_interval * (clustorizer_lon - 1)) &
                            (df_stations['station_longitude'] <= lon_min + lon_interval * clustorizer_lon),
                            'cluster_code'] = clustorizer_lat * 10 + clustorizer_lon

# stations repeat YoY. Take 2018 as the base as 2018 is the year covering all stations
df_stations_clusters = df_stations[['station_code', 'cluster_code']][df_stations['station_year'] == 2018]
##############################################################################################
# import trips
##############################################################################################
path = 'C:/Users/Ali/Documents/Insight/Bixi/Trips_2017_18'
all_files = glob.glob(path + "/*.csv")

listy = []
counter_f = 0
for f in all_files:
    print(f)
    col_names = ['start_date', 'start_station_code', 'end_date',
                 'end_station_code', 'duration_sec', 'is_member']
    df = pd.read_csv(f, skiprows=1, names=col_names)
    listy.append(df)

df_trips = pd.concat(listy, axis=0, ignore_index=True)

# Match trip and cluster info
df_trips_x = pd.merge(df_trips, df_stations_clusters[['station_code', 'cluster_code']],
                      left_on=['start_station_code'], right_on=['station_code'],
                      how='left')

# unpickle station-cluster & trips-cluster df: No need to run the prior
# section with these pickles
infile_df_trips_x = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_trips_x.pickle', 'rb')
df_trips_x = pickle.load(infile_df_trips_x)

infile_df_stations_clusters = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_stations_clusters.pickle', 'rb')
df_stations_clusters = pickle.load(infile_df_stations_clusters)


# Create time series features from datetime index
def create_features(df, label=None):

    df = df.copy()
    df['date'] = df.index
    df['hour'] = pd.DatetimeIndex(df['date']).hour
    df['dayofweek'] = pd.DatetimeIndex(df['date']).dayofweek
    df['quarter'] = pd.DatetimeIndex(df['date']).quarter
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['dayofyear'] = pd.DatetimeIndex(df['date']).dayofyear
    df['dayofmonth'] = pd.DatetimeIndex(df['date']).day
    df['weekofyear'] = pd.DatetimeIndex(df['date']).weekofyear
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
               'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
# ############################################################################################
# validate for given split date and cluster
# ############################################################################################


def validator(clusty, splitty):

    my_split_date = datetime.datetime.strptime(splitty, '%Y-%m-%d')
    my_split_next_date = my_split_date + timedelta(days=1)

    df_trips_x_one_cluster = df_trips_x[df_trips_x['cluster_code'] == clusty]
    df_trips_x_one_cluster['start_date'] = pd.to_datetime(df_trips_x_one_cluster['start_date'])
    df_trips_x_one_cluster = df_trips_x_one_cluster[df_trips_x_one_cluster['start_date'] >= '2017-01-01']

    # Get the number of pickups at mminute level.
    # When you resample on hour level they will be aggregated
    df_trips_x_one_cluster_agg_date = df_trips_x_one_cluster[['cluster_code', 'start_date']].groupby(['cluster_code', 'start_date']).size()
    df_trips_x_one_cluster_agg_date = df_trips_x_one_cluster_agg_date.reset_index()
    df_trips_x_one_cluster_agg_date.rename(columns={0: 'pickups'}, inplace=True)
    # Set the index as he datetime column to use the resample function below
    df_trips_x_one_cluster_agg_date = df_trips_x_one_cluster_agg_date.set_index('start_date')
    # Regroup pickups by X number of time units
    df_trips_x_one_cluster_agg_date_hourly = df_trips_x_one_cluster_agg_date.resample('H', closed='left').sum()
    # prepare data for the model
    pjme = df_trips_x_one_cluster_agg_date_hourly.copy()

    # detect outliers:
    pjme['z_pickups'] = np.abs(stats.zscore(df_trips_x_one_cluster_agg_date_hourly['pickups']))
    pjme['pickups'][pjme['z_pickups'] >= 3] = np.nan
    # print(np.where(z_pickups > 3)) # Gives outliers' rows

    pjme['pickups'] = pjme['pickups'].fillna(method="bfill")
    # Apply log transformation. That gives better results
    # Use log1p as some values are 0
    pjme['pickups'] = pjme['pickups'].apply(np.log1p)

    X, y = create_features(pjme, 'pickups')  # call the function above to create features

    features_and_target = pd.concat([X, y], axis=1)
    pjme = pjme.drop(['cluster_code'], axis=1)

    pjme_train = pjme.loc[pjme.index <= my_split_date].copy()
    pjme_test = pjme.loc[(pjme.index > my_split_date) &
                         (pjme.index <= my_split_next_date)].copy()

    #  setup and train your model
    model = Prophet()
    model.fit(pjme_train.reset_index().rename(columns={'start_date': 'ds', 'pickups': 'y'}))
    pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'start_date': 'ds'}))
    forecasts = pjme_test_fcst[['ds', 'yhat']].copy()
    forecasts['yhat_unlogged'] = round(np.expm1(forecasts['yhat']), 0)
    # Calculate R-squared
    print(r2_score(np.expm1(pjme_test['pickups']), np.expm1(forecasts['yhat'])))

# each time the vallidation function is called, the 24 hours following the split date is forecast.
# For a sound validation repeat this for 7 consecutive split dates and check the avg Rsquared
validator(23.0, '2018-09-01')

# ############################################################################################
# Once you get a validation, pickle your data for training model and application
# Validation may imply testing different number of clusters as well. That's why
# you should pickle station-cluster info after validation
# ############################################################################################
# pickle station info
outfile_df_stations = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_stations.pickle','wb')
pickle.dump(df_stations, outfile_df_stations)
outfile_df_stations.close()
# pickle station AND the associated cluster info
outfile_df_stations_clusters = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_stations_clusters.pickle',
                                    'wb')
pickle.dump(df_stations, outfile_df_stations_clusters)
outfile_df_stations_clusters.close()
# pickle your trip info along with the cluster of stations where pickup took place
pickle your trip-cluster info
outfile_df_trips_x = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_trips_x.pickle','wb')
pickle.dump(df_trips_x, outfile_df_trips_x)
outfile_df_trips_x.close()