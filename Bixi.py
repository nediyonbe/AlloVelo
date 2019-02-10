# import modules
import pandas as pd
import glob
import numpy as np
import datetime
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pickle
import os
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from scipy import stats
from datetime import timedelta

# unpickle station-cluster & trips-cluster df: No need to run the prior
# section with these pickles
infile_df_trips_x = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_trips_x.pickle','rb')
df_trips_x = pickle.load(infile_df_trips_x)

infile_df_stations_clusters = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_stations_clusters.pickle','rb')
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

# loop through clusters. This is because the resample function to follow
# works with single datetime index only. Having multiple clusters
# with same datetime prevents that
my_split_date = '2018-09-01'
my_split_date = datetime.datetime.strptime(my_split_date, '%Y-%m-%d')
my_split_next_date = my_split_date + timedelta(days=1)

for clusty in df_trips_x['cluster_code'].unique():

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
    # features_and_target.head()
    pjme = pjme.drop(['cluster_code'], axis=1)

    split_date = my_split_date
    pjme_train = pjme.loc[pjme.index <= split_date].copy()
    pjme_test = pjme.loc[(pjme.index > split_date) &
                         (pjme.index < my_split_next_date)].copy()

    #  setup and train your model
    model = Prophet()
    model.fit(pjme_train.reset_index().rename(columns={'start_date': 'ds', 'pickups': 'y'}))

    pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'start_date': 'ds'}))
    forecasts = pjme_test_fcst[['ds', 'yhat']].copy()
    forecasts['yhat_unlogged'] = round(np.expm1(forecasts['yhat']), 0)

    #  Pickle your model for later prediction use with the Graph_creator.py file
    outfile_prophet_model = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/models/prophet_model' + str(clusty) + '.pickle', 'wb')
    pickle.dump(model, outfile_prophet_model)
    outfile_prophet_model.close()
# End of Loop
