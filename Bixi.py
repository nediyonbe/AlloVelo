# In this code I look into the demand for the Bixi bike sharing in Montreal

#%%
print('test')
import pandas as pd
import glob

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
df_trips.describe()

# Import weather data
#%%
f_weather ='C:/Users/Ali/Documents/Insight/Bixi/Weather Data/eng-hourly-01012014-18-01-2019.csv' # use your path
col_names_weather = ['date_time_local', 'unixtime',	'pressure_station',	'pressure_sea',	'wind_dir',	'wind_dir_10s',	'wind_speed', 'wind_gust', 'relative_humidity', 'dew_point', 'temperature', 'windchill', 'humidex', 'visibility', 'health_index',	'cloud_cover_4', 'cloud_cover_8', 'cloud_cover_10', 'solar_radiation']
df_weather = pd.read_csv(f_weather, skiprows = 1, names = col_names_weather)
df_weather.describe()

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
df_trips['start_day_of_week'] = pd.DatetimeIndex(df_trips['start_date']).dayofweek
#%%
df_weather['weather_year'] = pd.DatetimeIndex(df_weather['date_time_local']).year
df_weather['weather_month'] = pd.DatetimeIndex(df_weather['date_time_local']).month
df_weather['weather_day'] = pd.DatetimeIndex(df_weather['date_time_local']).day
df_weather['weather_hour'] = pd.DatetimeIndex(df_weather['date_time_local']).hour
df_weather['weather_day_of_week'] = pd.DatetimeIndex(df_weather['date_time_local']).dayofweek

#Create the coordianates tuple for mapping
#%%
df_trips.describe()
df_trips.dtypes

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
#%%
df_stations = pd.concat(listy_stations, axis = 0, ignore_index = True)
df_stations.describe()

#check import
#%%
df_stations.groupby('station_year').size()

#%%
#Map the locations of stations YoY. This will let you see the change of station allocation across neighborhoods
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

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
        style = 'dark'
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
#%%
#plotly.plotly.iplot(data, filename = 'C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/Templates/charty.html')


# for basic analyses:
# for_plot = df_trips.groupby('start_month').size()
# for_plot
