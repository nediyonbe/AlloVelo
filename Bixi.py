# In this code I look into the demand for the Bixi bike sharing in Montreal

#%%
print('test')
import pandas as pd
import glob
import numpy as np
import datetime

# Import bicyle use data
#%%
path ='C:/Users/Ali/Documents/Insight/Bixi/Trips' # use your path
all_files = glob.glob(path + "/*.csv")

#%%
listy = []
counter_f = 0
for f in all_files:
    print(f)
    col_names = ['start_date', 'start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member']
    df = pd.read_csv(f, skiprows = 1, names = col_names)
    listy.append(df)

#%%
df_trips = pd.concat(listy, axis = 0, ignore_index = True)

# Import weather data
#%%
f_weather ='C:/Users/Ali/Documents/Insight/Bixi/Weather Data/eng-hourly-01012014-18-01-2019_modified.csv' # use your path
col_names_weather = ['date_time_local', 'unixtime',	'pressure_station',	'pressure_sea',	'wind_dir',	'wind_dir_10s',	'wind_speed', 'wind_gust', 'relative_humidity', 'dew_point', 'temperature', 'windchill', 'humidex', 'visibility', 'health_index',	'cloud_cover_4', 'cloud_cover_8', 'cloud_cover_10', 'solar_radiation', 'humidex_filled', 'cloud_cover_8_filled', 'wind_dir_filled']
df_weather = pd.read_csv(f_weather, skiprows = 1, names = col_names_weather)

#Check data types, datetime fields may not be
#%%
df_trips.dtypes
df_weather.dtypes
df_weather.head()
#Set related fields to datetime
#%%
df_trips['start_date'] = pd.to_datetime(df_trips['start_date'])
df_trips['end_date'] = pd.to_datetime(df_trips['end_date'])

#Get year and month info in separate fields. This can be useful for analyses
#%%
df_trips['start_year'] = pd.DatetimeIndex(df_trips['start_date']).year
df_trips['start_month'] = pd.DatetimeIndex(df_trips['start_date']).month
df_trips['start_day'] = pd.DatetimeIndex(df_trips['start_date']).day
df_trips['start_hour'] = pd.DatetimeIndex(df_trips['start_date']).hour
df_trips['start_day_of_week'] = pd.DatetimeIndex(df_trips['start_date']).dayofweek + 1 #Monday is 0, Tue is 1
df_trips['start_just_date'] = pd.DatetimeIndex(df_trips['start_date']).date
df_trips['start_just_date_hr'] = df_trips['start_year'].map(str) + df_trips['start_month'].apply(lambda x: '{0:0>2}'.format(x)).map(str) + df_trips['start_day'].apply(lambda x: '{0:0>2}'.format(x)).map(str) +df_trips['start_hour'].apply(lambda x: '{0:0>2}'.format(x)).map(str) 
df_trips['start_just_date_hr'] = pd.to_numeric(df_trips['start_just_date_hr'])
df_weather['weather_year'] = pd.DatetimeIndex(df_weather['date_time_local']).year
df_weather['weather_month'] = pd.DatetimeIndex(df_weather['date_time_local']).month
df_weather['weather_day'] = pd.DatetimeIndex(df_weather['date_time_local']).day
df_weather['weather_hour'] = pd.DatetimeIndex(df_weather['date_time_local']).hour
df_weather['weather_day_of_week'] = pd.DatetimeIndex(df_weather['date_time_local']).dayofweek + 1 #Monday is 0, Tue is 1
df_weather['weather_just_date'] = pd.DatetimeIndex(df_weather['date_time_local']).date
df_weather['weather_just_date_hr'] = df_weather['weather_year'].map(str) + df_weather['weather_month'].apply(lambda x: '{0:0>2}'.format(x)).map(str) + df_weather['weather_day'].apply(lambda x: '{0:0>2}'.format(x)).map(str) +df_weather['weather_hour'].apply(lambda x: '{0:0>2}'.format(x)).map(str) 
df_weather['weather_just_date_hr'] = pd.to_numeric(df_weather['weather_just_date_hr'])
#Create the coordianates tuple for mapping
#%%
df_weather.info()

#Import stations and their coordinates for mapping
#%%
path_stations ='C:/Users/Ali/Documents/Insight/Bixi/Stations' # use your path
all_files_stations = glob.glob(path_stations + "/*.csv")
#%%
listy_stations = []
counter_f_stations = 0
for f in all_files_stations:
    print(f)
    col_names_stations = ['station_code', 'station_name', 'station_latitude', 'station_longitude', 'station_year']
    df = pd.read_csv(f, skiprows = 1, names = col_names_stations)
    listy_stations.append(df)
df_stations = pd.concat(listy_stations, axis = 0, ignore_index = True)
#%%
df_stations.head()

#check import   
#%%
df_stations.groupby('station_year').size()

#%%
# Define clusters: To be done in a detailed fashion later
# Pickup and drops will be regrouped by clusters
# For test purposes allocate stations on a 10x10 matrix limited by the following coordinates:
# N: 45.56, S:45.42, W:-73.68, E:-73.49
lat_max = df_stations['station_latitude'].max()
lat_min = df_stations['station_latitude'].min()
lon_max = df_stations['station_longitude'].max()
lon_min = df_stations['station_longitude'].min()
num_clusters = 5
lat_interval = (lat_max - lat_min) / num_clusters
lon_interval = (lon_max - lon_min) / num_clusters
#%%
for clustorizer_lat in range(1,num_clusters+1):
    print(clustorizer_lat)
    for clustorizer_lon in range(1,num_clusters+1):
            df_stations.loc[(df_stations['station_latitude'] >= lat_min + lat_interval * (clustorizer_lat - 1)) &
                    (df_stations['station_latitude'] <= lat_min + lat_interval * clustorizer_lat) &
                    (df_stations['station_longitude'] >= lon_min + lon_interval * (clustorizer_lon - 1)) &
                    (df_stations['station_longitude'] <= lon_min + lon_interval * clustorizer_lon),    
    'cluster_code'] = clustorizer_lat * 10 + clustorizer_lon 
#df_stations.describe()
# print(lat_max)
# print(lat_min)
# print(lon_max)
# print(lon_min)
# print(lon_interval)
#%%
df_stations[df_stations['cluster_code'].isnull()].head()
#%%
df_stations.groupby('cluster_code').size()

#%%
#stations repeat YoY. Take 2018 as the base as some stations slightly change location causing cluster changes when at the border
df_stations_clusters = df_stations[['station_code', 'cluster_code']][df_stations['station_year'] == 2018]
#%%
df_stations_clusters.head()
#%%
df_trips.head()
#%%
# Prepare your df that will be input to the algorithm: To be done in a detailed fashion later
# 1. Join df_trips and df_stations over station_code to get cluster_code
# 2. Aggregate df_trips over start_just_date and cluster_code with pickup count
# 3. Aggregate df_weather over daweather_just_date with max temp, max wind speed
# 4. Join df_trips and df_weather over start_just_date = weather_just_date to get daily max temp and wind
#Step1&2:
#df_trips['start_just_date'].apply(str) #grouping by date filed crashed
df_trips_agg_date = df_trips[['start_station_code','start_just_date_hr']].groupby(['start_station_code','start_just_date_hr']).size()
#%%
df_trips_agg_date.head()

#%%
df_trips_agg_date = df_trips_agg_date.reset_index()
#%%
df_trips_agg_date.rename(columns={0:'pickups'}, inplace=True)
#%%
df_trips_agg_date['pickups'].sum()
#%%
df_trips_agg_date.head() 
#%%
df_trips_agg_date_cluster = pd.merge(df_trips_agg_date, df_stations_clusters[['station_code', 'cluster_code']], 
                            left_on=['start_station_code'], right_on=['station_code'], 
                            how='left')    
#df_trips_agg_date.to_csv('C:/Users/Ali/Desktop/xyztripsagg.csv', index = False)     
#df_stations.to_csv('C:/Users/Ali/Desktop/xyzstation.csv', index = False)                                
#df_trips_agg_date_cluster.info()    
#df_trips_agg_date_cluster.sort_values(by=['start_station_code'])                    

#Step3
#%%
df_trips_agg_date_cluster.info() #6348847 entries regrouped by hour
#%%
df_weather.info()
#%%
df_weather.head()
#%%
df_trips_agg_date_cluster[df_trips_agg_date_cluster['cluster_code'].isnull()].head()
#%%
df_weather_agg_date = df_weather[['weather_just_date_hr', 'weather_month', 'weather_day_of_week','weather_hour','weather_year','temperature','wind_speed','relative_humidity','wind_gust', 'cloud_cover_8_filled','humidex_filled','wind_dir_10s']].groupby('weather_just_date_hr').max()
df_weather_agg_date = df_weather_agg_date.reset_index()
#df_weather_agg_date.rename(columns={0:'temperature'}, inplace=True)

# check solar radiation NULL values. The info command gives non-null for the field but there are nulls
#%%
df_weather_agg_date['wind_gust'] = df_weather_agg_date['wind_gust'].fillna(0)
df_weather_agg_date.head()

#%%
# Step4:
df_trips_agg_date_cluster_temp = pd.merge(df_trips_agg_date_cluster, df_weather_agg_date, 
                            left_on=['start_just_date_hr'], right_on=['weather_just_date_hr'], 
                            how='left')
#%%                      
#df_trips_agg_date_cluster_temp.head()
df_trips_agg_date_cluster_temp.groupby('weather_year').size()
#%%  
df_trips_agg_date_cluster_temp = df_trips_agg_date_cluster_temp[df_trips_agg_date_cluster_temp['weather_year'] == 2018]
#%%  
df_trips_agg_date_cluster_temp.info()

#Transform cyclical data
df_trips_agg_date_cluster_temp['weather_month_sin']  = np.sin(df_trips_agg_date_cluster_temp['weather_month'] * (2.*np.pi/24))
df_trips_agg_date_cluster_temp['weather_month_cos']  = np.cos(df_trips_agg_date_cluster_temp['weather_month'] * (2.*np.pi/24))

df_trips_agg_date_cluster_temp['weather_hour_sin']  = np.sin((df_trips_agg_date_cluster_temp['weather_hour'] - 1) * (2.*np.pi/12))
df_trips_agg_date_cluster_temp['weather_hour_cos']  = np.cos((df_trips_agg_date_cluster_temp['weather_hour'] - 1) * (2.*np.pi/12))

dfy = pd.get_dummies(df_trips_agg_date_cluster_temp['cluster_code'])
dfy = dfy.reset_index()
#dfy.iloc[1000000]
df_trips_agg_date_cluster_temp_eng = pd.concat([df_trips_agg_date_cluster_temp, dfy], axis = 1)
df_trips_agg_date_cluster_temp_eng = df_trips_agg_date_cluster_temp_eng.reset_index()
df_trips_agg_date_cluster_temp_eng.head(-5)

# dfz.info()
# dfz[['cluster_code', 11.0, 12.0, 13.0, 54.0, 21.0]][df_trips_agg_date_cluster_temp['cluster_code'] == 54].describe()
#%%
from sklearn import preprocessing
target = df_trips_agg_date_cluster_temp_eng['pickups']
#%%
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df_trips_agg_date_cluster_temp_eng[['temperature','wind_speed','relative_humidity','wind_gust', 'cloud_cover_8_filled','humidex_filled','wind_dir_10s']])
daty = pd.DataFrame(x_scaled)
daty.rename(columns={0: 'temperature', 1: 'wind_speed', 2:'relative_humidity', 3:'wind_gust', 4:'cloud_cover_8_filled', 5:'humidex_filled', 6:'wind_dir_10s' }, inplace=True)
df_trips_agg_date_cluster_temp.head()
#data = preprocessing.normalize(df_trips_agg_date_cluster_temp_eng[['temperature','wind_speed','relative_humidity','wind_gust', 'cloud_cover_8_filled','humidex_filled','wind_dir_10s']])
datz = pd.merge(daty, df_trips_agg_date_cluster_temp_eng[['cluster_code', 'weather_month_sin', 'weather_month_cos', 'weather_hour_sin', 'weather_hour_cos']], left_index=True, right_index=True, how = 'inner')
datz = datz.reset_index()
data = pd.merge(datz, dfy,  left_index=True, right_index=True, how = 'inner')
#CHECK: data[['cluster_code', 11.0, 12.0, 13.0, 54.0, 21.0]][data['cluster_code'] == 12].describe()
#data['weather_hour_sin'].describe()
#df_trips_agg_date_cluster_temp_eng['weather_hour_sin'].describe()
X_train.shape
x_scaled.shape
df_trips_agg_date_cluster_temp_eng.info()
daty.describe()
daty.head(-5)
daty.info()
data.info()
dfy.head(-5)
data.head(-5)
target.shape
#Knn regression explains 28% variation, better than the regular regression w/ 8%
#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
#import KNeighborsRegressor
#%%
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state = 0)
#%%
print(datetime.datetime.now().time())
knnreg = KNeighborsRegressor(n_neighbors = 31).fit(X_train, y_train)
print(datetime.datetime.now().time())
print(knnreg.score(X_test, y_test)) #Get the R-squared
#%%
y_predict_output = knnreg.predict(X_test)
#%%
X_test.info()
#%%
y_predict_output.size
#%%
df_trips_agg_date_cluster_temp.info()
# dfx = df_trips_agg_date_cluster_temp
# dfx['start_just_date'] = pd.to_datetime(df_trips_agg_date_cluster_temp['start_just_date'])
# dfx[dfx['start_just_date'] == '2017-05-01'].head(10)
#%%
df_trips_agg_date_cluster.describe()
#%%
df_weather_agg_date.describe()
#%%
#Map the locations of stations YoY. This will let you see the change of station allocation across neighborhoods
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

#%%
df_stations.head()

# setting user, api key and access token
#%%
plotly.tools.set_credentials_file(username='nediyonbe', api_key='pBiKl18jmZiSZU8BzwDY')
mapbox_access_token = 'pk.eyJ1IjoibmVkaXlvbmJlIiwiYSI6ImNqcjZreXEzZDE0Yzg0OHBuaTdrMTdmcWEifQ.DFTnN07buUlCze5IM-S_tg'
#Create your data feeding the map
#%%
anno_types = [2014, 2015, 2016, 2017, 2018]
data = []
for anno in anno_types:
    station_data = dict(
            lat = df_stations.loc[df_stations['station_year'] == anno,'station_latitude'],
            lon = df_stations.loc[df_stations['station_year'] == anno,'station_longitude'],
            name = anno,
            text = df_stations.loc[df_stations['station_year'] == anno, 'station_code'].apply(str) + ' ' + df_stations.loc[df_stations['station_year'] == anno, 'station_name'].apply(str),
            
            hoverinfo = 'text',
            marker = dict(size = 8, opacity = 0.5),
            type = 'scattermapbox'
        )
    data.append(station_data)
#Create your map layout
#%%
layout = dict(
    height = 800,
    width = 800,
    # top, bottom, left and right margins
    margin = dict(t = 0, b = 0, l = 0, r = 0),
    font = dict(color = '#FFFFFF', size = 11),
    paper_bgcolor = '#000000',
    mapbox = dict(
        # here you need the token from Mapbox
        accesstoken = 'pk.eyJ1IjoibmVkaXlvbmJlIiwiYSI6ImNqcjZreXEzZDE0Yzg0OHBuaTdrMTdmcWEifQ.DFTnN07buUlCze5IM-S_tg',
        bearing = 0,
        # where we want the map to be centered i.e. Place des Arts (MTL)
        center = dict(
            lat = 45.51,
            lon = -73.57
        ),
        # we want the map to be "parallel" to our screen, with no angle
        pitch = 0,
        # default level of zoom
        zoom = 11,
        # default map style
        style = 'outdoors'
    )
)
#Create your drop downs for the map
#%%
updatemenus=list([
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
        direction = 'up',
        # here I specify where I want to place this drop-down on the map
        x = 0.75,
        xanchor = 'left',
        y = 0.05,
        yanchor = 'bottom',
              # specify font size and colors
        bgcolor = '#000000',
        bordercolor = '#FFFFFF',
        font = dict(size=11)
    ),    
    # drop-down 2: select type of storm event to visualize
    dict(
         # for each button I specify which dictionaries of my data list I want to visualize. Remember I have 7 different
         # types of storms but I have 8 options: the first will show all of them, while from the second to the last option, only
         # one type at the time will be shown on the map
         buttons=list([
            dict(label = '2014',
                 method = 'update',
                 args = [{'visible': [True, False, False, False, False]}]),
            dict(label = '2015',
                 method = 'update',
                 args = [{'visible': [False, True, False, False, False]}]),
             dict(label = '2016',
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False]}]),
             dict(label = '2017',
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False]}]),
             dict(label = '2018',
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True]}])          
        ]),
        # direction where the drop-down expands when opened
        direction = 'down',
        # positional arguments
        x = 0.01,
        xanchor = 'left',
        y = 0.99,
        yanchor = 'bottom',
        # fonts and border
        bgcolor = '#000000',
        bordercolor = '#FFFFFF',
        font = dict(size=11)
    )
])
# assign the list of dictionaries to the layout dictionary
layout['updatemenus'] = updatemenus
#%%
layout['title'] = 'Bixi Stations over Years'
#%%
figure = dict(data = data, layout = layout)
py.iplot(figure, filename = 'Bixi_Stations_over_Years_File')
# for basic analyses:
# for_plot = df_trips.groupby('start_month').size()
# for_plot
