import pandas as pd
import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from flask import flash, redirect, url_for, render_template
from fbprophet import Prophet
import pickle
import os
from datetime import timedelta
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

# Unpickle station-cluster & trips-cluster df # No need to run the prior section with these pickles
infile_df_trips_x = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_trips_x.pickle','rb')
df_trips_x = pickle.load(infile_df_trips_x)

infile_df_stations_clusters = open('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/df_stations_clusters.pickle','rb')
df_stations_clusters = pickle.load(infile_df_stations_clusters)


# ###################################################################
# pick up your previously trained model and make the prediction for
# the requested date
# ###################################################################
# This function is called by the application python file
def pickup_estimator(mydate):
    mydate = datetime.datetime.strptime(mydate, '%Y-%m-%d')
    mynextdate = mydate + timedelta(days=1)
    mydate_dayofweek = mydate.weekday()
    mydate_month = mydate.month
    # loop for Every File in the Folder Where Model Pickle of Each Cluster is stored
    model_directory = 'C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/models'
    forecasts_results_all_clusters = pd.DataFrame(columns=['cluster_code', 'yhat'])
    for filename in os.listdir(model_directory):
        # get the cluster number which is a float:
        clusty = filename[-11:-7]  # Note that this works as long as clusters are between 10 and 99 with one decimal point

        infile_outfile_prophet_model = open(model_directory + '/prophet_model' + clusty + '.pickle', 'rb')
        modelP = pickle.load(infile_outfile_prophet_model)

        # create your time period that you will predict
        # the function creates record for midnight of both start and end dates. Drop the very last record
        forecast_period = pd.date_range(mydate, mynextdate, freq='H')[:24]
        df_forecast_period = pd.DataFrame(data=forecast_period)
        # prophet uses ds column. Change the name
        df_forecast_period.columns = ['ds']
        # make the prediction for the cluster in loop
        forecasts_results = modelP.predict(df=df_forecast_period)
        forecasts_results_temp = forecasts_results[['ds', 'yhat']]
        forecasts_results_temp['cluster_code'] = clusty
        # unite with other clusters' predictions
        forecasts_results_all_clusters = pd.concat([forecasts_results_all_clusters, forecasts_results_temp], axis=0)
    # End of loop
    # ###################################################################
    # Reformat your prediction table according to plotly requirements
    # ###################################################################
    # convert cluster code to number
    forecasts_results_all_clusters['cluster_code'] = pd.to_numeric(forecasts_results_all_clusters['cluster_code'])
    # untransform log values
    forecasts_results_all_clusters['yhat_unlogged'] = round(np.expm1(forecasts_results_all_clusters['yhat']), 0)
    # you will need an hour column for retrieving the demand for the related hour's demand
    forecasts_results_all_clusters['hour'] = pd.DatetimeIndex(forecasts_results_all_clusters['ds']).hour

    df_trips_x_cluster_min_demand = df_trips_x[['cluster_code', 'start_date']].groupby(['cluster_code', 'start_date']).size()

    # set the start date as index to regroup by hour
    df_trips_x_cluster_min_demand = df_trips_x_cluster_min_demand.reset_index()
    df_trips_x_cluster_min_demand['hour'] = pd.DatetimeIndex(df_trips_x_cluster_min_demand['start_date']).hour
    df_trips_x_cluster_min_demand['dayofweek'] = pd.DatetimeIndex(df_trips_x_cluster_min_demand['start_date']).dayofweek
    df_trips_x_cluster_min_demand['month'] = pd.DatetimeIndex(df_trips_x_cluster_min_demand['start_date']).month
    df_trips_x_cluster_min_demand = df_trips_x_cluster_min_demand.rename(columns={0: 'pickups_avg'})
    # for comparison betweent he prediction and the hitorical averages,
    # calculate the average demand of the same i) day of week ii) month
    # for each i) cluster ii) month iii) day of week iv) hour
    df_trips_x_cluster_hour_demand = df_trips_x_cluster_min_demand[['cluster_code', 'month', 'dayofweek', 'hour', 'pickups_avg']][(df_trips_x_cluster_min_demand['month'] == mydate_month) & (df_trips_x_cluster_min_demand['dayofweek'] == mydate_dayofweek)].groupby(['cluster_code', 'month', 'dayofweek', 'hour']).agg({'pickups_avg': np.mean})
    df_trips_x_cluster_hour_demand = df_trips_x_cluster_hour_demand.reset_index()

    show_this = pd.merge(forecasts_results_all_clusters, df_trips_x_cluster_hour_demand, 
                         left_on=['cluster_code', 'hour'],
                         right_on=['cluster_code', 'hour'],
                         how='left')
    # the right table in the join does not have pickup for every hour.
    # They show up as null. Make them zero
    show_this['pickups_avg'] = show_this['pickups_avg'].fillna(0)

    # the right table in the join was already filtered
    # for the day of the week and month in question. No need to keep those columns
    show_this = show_this.drop(columns=['yhat', 'month', 'dayofweek'])
    # calculate the difference between the prediction and
    # the avg of the same hr/day of week/ month for the same cluster
    show_this['comparison_to_avg_absolute'] = show_this['yhat_unlogged'] - show_this['pickups_avg']

    # MinMax scale the comparison to avg absolute column to see
    # at what part of the city AND at what time there will be the highest demand shift
    # the scaler of scikit does not work properly with single column. Calculate manually instead
    maxy = show_this['comparison_to_avg_absolute'].max()
    miny = show_this['comparison_to_avg_absolute'].min()
    show_this['comparison_to_avg_absolute_scaled'] = show_this['comparison_to_avg_absolute'].apply(lambda x: (x - miny)/(maxy-miny))
    # ###################################################################
    # create map
    # ###################################################################
    # set user, api key and access token for the mapbox and plotly
    plotly.tools.set_credentials_file(username='nediyonbe', api_key='pBiKl18jmZiSZU8BzwDY')
    mapbox_access_token = 'pk.eyJ1IjoibmVkaXlvbmJlIiwiYSI6ImNqcjZreXEzZDE0Yzg0OHBuaTdrMTdmcWEifQ.DFTnN07buUlCze5IM-S_tg'
    # create a unique list of hours to loop through for the map
    hours = show_this['hour'].unique()
    # use 2018 clusters as it covers all stations
    df_stations_clusters_2018 = df_stations_clusters[df_stations_clusters['station_year'] == 2018]

    show_this_expanded_stations = pd.merge(df_stations_clusters_2018, show_this,
                                           left_on=['cluster_code'],
                                           right_on=['cluster_code'], how='left')

    show_this_expanded_stations['comparison_to_avg_absolute_scaled'] = pd.to_numeric(show_this_expanded_stations['comparison_to_avg_absolute_scaled'])

    show_this_expanded_stations.loc[show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.1, 'RGB'] = 'RGB(88, 214, 141)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.1) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.2), 'RGB'] = 'RGB(0, 255, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.2) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.3), 'RGB'] = 'RGB(141, 255, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.3) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.4), 'RGB'] = 'RGB(212, 255, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.4) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.5), 'RGB'] = 'RGB(255, 246, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.5) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.6), 'RGB'] = 'RGB(255, 211, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.6) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.7), 'RGB'] = 'RGB(255, 175, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.7) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.8), 'RGB'] = 'RGB(255, 123, 0)'
    show_this_expanded_stations.loc[(show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.8) & (show_this_expanded_stations['comparison_to_avg_absolute_scaled'] < 0.9), 'RGB'] = 'RGB(255, 70, 0)'
    show_this_expanded_stations.loc[show_this_expanded_stations['comparison_to_avg_absolute_scaled'] >= 0.9, 'RGB'] = 'RGB(255, 0, 0)'

    # create a dictionary of lists where each hour (key) corresponds to
    # a list of TRUEs for the plotly layout. For h = 0 the first element of
    # the list will be TRUE the remaining 23 will be FALSE
    button_visibility_arg = {}
    for h in range(0, 24):
        button_visibility_arg[h] = [False] * 24
        button_visibility_arg[h][h] = True
    data = []  # create the list that will hold the data feeding the plotly map
    for h in hours:
        station_data = dict(
            lat=df_stations_clusters_2018['station_latitude'],
            lon=df_stations_clusters_2018['station_longitude'],
            name=str(h),
            text=show_this_expanded_stations['station_code'][show_this_expanded_stations['hour'] == h].apply(str) + \
            ' ' + show_this_expanded_stations['cluster_code'][show_this_expanded_stations['hour'] == h].apply(str) + \
            ' ' + show_this_expanded_stations['yhat_unlogged'][show_this_expanded_stations['hour'] == h].apply(str) + \
            ' ' + show_this_expanded_stations['comparison_to_avg_absolute_scaled'][(show_this_expanded_stations['hour'] == h)].apply(lambda x: str(round(x, 1))),           
            hoverinfo='text',
            marker=dict(size=8, opacity=0.5, color=show_this_expanded_stations['RGB'][(show_this_expanded_stations['hour'] == h)]),
            type='scattermapbox'
            )
        data.append(station_data)

    # Create your map layout
    layout = dict(
        height=800,
        width=800,
        # top, bottom, left and right margins
        margin=dict(t=0, b=0, l=0, r=0),
        font=dict(color='#FFFFFF', size=11),
        paper_bgcolor='#000000',
        mapbox=dict(
            # here you need the token from Mapbox
            accesstoken='pk.eyJ1IjoibmVkaXlvbmJlIiwiYSI6ImNqcjZreXEzZDE0Yzg0OHBuaTdrMTdmcWEifQ.DFTnN07buUlCze5IM-S_tg',
            bearing=0,
            # where we want the map to be centered i.e. Place des Arts (MTL)
            center=dict(
                lat=45.51,
                lon=-73.57
            ),
            # we want the map to be "parallel" to our screen, with no angle
            pitch=0,
            # default level of zoom
            zoom=11,
            # default map style: Dark is more visible with colored scatter points
            style='dark'
        )
    )
    # Create your drop downs for the map
    updatemenus = list([
        # drop-down 1: map styles menu
        # buttons containes as many dictionaries as many alternative map styles I want to offer
        dict(
            buttons=list([
                dict(
                    args=['mapbox.style', 'dark'],
                    label='Dark',
                    method='relayout'
                ),                    
                dict(
                    args=['mapbox.style', 'light'],
                    label='Light',
                    method='relayout'
                ),
                dict(
                    args=['mapbox.style', 'outdoors'],
                    label='Outdoors',
                    method='relayout'
                ),
                dict(
                    args=['mapbox.style', 'satellite-streets'],
                    label='Satellite with Streets',
                    method='relayout'
                )                    
            ]),
            # direction where I want the menu to expand when I click on it
            direction='up',
            # here I specify where I want to place this drop-down on the map
            x=0.75,
            xanchor='left',
            y=0.05,
            yanchor='bottom',
            #  specify font size and colors
            bgcolor='#000000',
            bordercolor='#FFFFFF',
            font=dict(size=11)
        ),
        # drop-down 2: select hour to visualize
        dict(
            # for each button I specify which dictionaries of my data list I want to visualize
            buttons=list([
                dict(label='0',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[0]}]),
                dict(label = '1',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[1]}]),
                dict(label = '2',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[2]}]),
                dict(label = '3',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[3]}]),
                dict(label = '4',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[4]}]),
                dict(label = '5',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[5]}]),
                dict(label = '6',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[6]}]),
                dict(label = '7',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[7]}]),
                dict(label = '8',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[8]}]),
                dict(label = '9',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[9]}]),
                dict(label = '10',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[10]}]),
                dict(label = '11',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[11]}]),
                dict(label = '12',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[12]}]),
                dict(label = '13',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[13]}]),
                dict(label = '14',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[14]}]),
                dict(label = '15',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[15]}]),
                dict(label = '16',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[16]}]),
                dict(label = '17',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[17]}]),
                dict(label = '18',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[18]}]),
                dict(label = '19',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[19]}]),
                dict(label = '20',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[20]}]),
                dict(label = '21',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[21]}]),
                dict(label = '22',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[22]}]),
                dict(label = '23',
                    method = 'update',
                    args = [{'visible': button_visibility_arg[23]}])
            ]),
            # direction where the drop-down expands when opened
            direction='down',
            # positional arguments
            x=0.01,
            xanchor='left',
            y=0.99,
            yanchor='bottom',
            # fonts and border
            bgcolor='#000000',
            bordercolor='#FFFFFF',
            font=dict(size=11)
        )
    ])
    # assign the list of dictionaries to the layout dictionary
    layout['updatemenus'] = updatemenus
    layout['title'] = 'Bixi Pickup Forecast' + ' ' + str(mydate.date())
    figure = dict(data = data, layout = layout)
    py.iplot(figure, filename = 'Bixi_Pickup_Demands')