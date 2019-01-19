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

#%%
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
#%%
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



#%%
for_plot = df_trips.groupby('start_month').size()
for_plot

#%%
import seaborn as sns
ax = sns.barplot(data = for_plot)

#%%
label_x = ["Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]
ax = x.plot.bar(x='label_x', y = 'for_plot', rot = 0)