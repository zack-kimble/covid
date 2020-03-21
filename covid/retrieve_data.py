#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:25:48 2020

@author: zack
"""

#!/usr/bin/env python
# coding: utf-8


import requests
import pandas as pd
import io
import numpy as np

# Uses WorldWeatherOnline package: https://github.com/ekapope/WorldWeatherOnline. I am have made a bug fix and added caching.
# I don't think the bug will effect the data unless you run on the last day of the month. The caching is very useful for limitting calls though
# You can clone my fork and use it in the meantime: https://github.com/zack-kimble/WorldWeatherOnline
# or install directly: pip install -e git+https://github.com/zack-kimble/WorldWeatherOnline.git#egg=wwo_hist


from wwo_hist import retrieve_hist_data
import socket #need this for socket timeout from requests. Odd
import urllib

config = dict(
use_local = False,
index_col = ['Country/Region', 'Province/State', 'Lat', 'Long', 'date'],
weather_api_key = 'api_key_here' ,
data_dir = 'data'
)


def get_melt_clean(url,value_name,use_local):   
    file_name = url.split(sep='/')[-1]
    if use_local:
        df = pd.read_csv(config['data_dir']+'/'+file_name)
    else:
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv(config['data_dir']+'/'+file_name, index=False)
    id_cols = ['Country/Region','Province/State', 'Lat', 'Long']
    df = df.melt(id_vars=id_cols, var_name='date', value_name=value_name)
    df['date'] = pd.to_datetime(df.date)
    df.fillna({'Province/State': 'all'}, inplace=True) #needed to make groupby work correctly
    return df

def filter_countries(full_df, min_deaths, min_cases):
    include_countries = full_df[np.any((full_df.deaths >= min_deaths, full_df.cases >= min_cases),0)]['Country/Region'].unique()
    filtered_df = full_df[full_df['Country/Region'].apply(lambda x: x in include_countries)].copy()
    return filtered_df

def filter_provinces(full_df, min_deaths, min_cases):
    include_provinces = full_df[np.any((full_df.deaths >= min_deaths, full_df.cases >= min_cases),0)]['Province/State'].unique()
    include_countries = full_df[np.any((full_df.deaths >= min_deaths, full_df.cases >= min_cases), 0)]['Country/Region'].unique()
    province_idx = full_df['Province/State'].apply(lambda x: x in include_provinces)
    country_idx = full_df['Country/Region'].apply(lambda x: x in include_countries)
    filtered_df = full_df[np.any(np.array([country_idx, province_idx]), axis=0)].copy()

    return filtered_df


def retrieve_data(config=config):

    deaths_df = get_melt_clean(url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
                               value_name='deaths',
                               use_local= config['use_local']
                               )
    cases_df = get_melt_clean(url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
                              value_name='cases',
                              use_local= config['use_local'])
    recovered_df = get_melt_clean(url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv',
                              value_name='recoveries',
                              use_local= config['use_local'])

    join_cols = config['index_col']
    full_df = pd.merge(cases_df, deaths_df, on=join_cols)
    full_df = pd.merge(full_df, recovered_df, on=join_cols)

    # countries with at least 100 cases or 1 death
    filtered_df = filter_provinces(full_df,1,50)
    filtered_df['new_deaths'] = filtered_df.groupby(['Country/Region','Province/State'])['deaths'].diff().clip(lower=0)
    filtered_df['new_cases'] = filtered_df.groupby(['Country/Region','Province/State'])['cases'].diff().clip(lower=0)

    #calculate active cases
    filtered_df['active_cases'] = filtered_df['cases'] - filtered_df['deaths'] - filtered_df['recoveries']
    filtered_df['new_cases_as_percent_of_active'] = filtered_df['new_cases'] / filtered_df['active_cases']

    #filtered_df = add_dates_to_index(filtered_df, -14, join_cols)
    filtered_df = filtered_df.set_index(join_cols).select_dtypes(include='number').fillna(0)

    #chang index to lat/long and date for use with weather
    filtered_df.reset_index(['Country/Region','Province/State'], inplace=True)

    #load local weather cache and check of missing location/dates
    try:
        hist_weather_df_cache = pd.read_csv(config['data_dir']+'/wwo_cache.csv')
        hist_weather_df_cache['date'] = pd.to_datetime(hist_weather_df_cache['date'])
        hist_weather_df_cache = hist_weather_df_cache.set_index(['Lat','Long','date'])

    except FileNotFoundError:
        hist_weather_df_cache = pd.DataFrame()

    #get unique missing lat/long date tuples
    missing_indexes = filtered_df.index.difference(hist_weather_df_cache.index)

    if len(missing_indexes) > 0:
        missing_val_df = filtered_df.loc[missing_indexes]
        missing_val_agg = missing_val_df.reset_index().groupby(['Lat','Long']).aggregate(start_date=('date','min'),end_date=('date','max')).reset_index()
        missing_val_agg['lat_long_string'] = [str(x) + ','+ str(y) for x, y in zip(missing_val_agg['Lat'], missing_val_agg['Long'])]

        #retrieve missing weather data

        hist_weather_list = []

        for row in missing_val_agg.itertuples():
            try:
                hist_weather_list_loc = retrieve_hist_data(
                                                api_key = config['weather_api_key'],
                                                location_list=[row.lat_long_string],
                                                start_date=row.start_date,
                                                end_date=row.end_date,
                                                frequency=24,
                                                location_label = False,
                                                export_csv = False,
                                                store_df = True,
                                                response_cache_path='woo_cache')

                hist_weather_list.extend(hist_weather_list_loc)
            except requests.HTTPError:
                print("exceded daily request limit, saving retrieved and exiting")
                break
            except socket.timeout:
                print("timed out, saving retrieved and exiting")
                break
            except urllib.error.URLError:
                print('network issue, saving retrieved and exiting')
                break
#TODO: need to check cached responses even if there's a current connection failure. There's some mixed up dependencies though. Caches are checked inside wwo_hist, but this is feeding requests one by one
        try:
            hist_weather_df_new = pd.concat(hist_weather_list)
            hist_weather_df_new['Lat'] = hist_weather_df_new['location'].apply(lambda x: x.split(',')[0]).astype('float')
            hist_weather_df_new['Long'] = hist_weather_df_new['location'].apply(lambda x: x.split(',')[1]).astype('float')
            hist_weather_df_new.rename(columns={'date_time':'date'},inplace=True)
            hist_weather_df_new.set_index(['Lat','Long','date'],inplace=True)
            hist_weather_df = pd.concat([hist_weather_df_cache, hist_weather_df_new], verify_integrity=True)
            hist_weather_df.to_csv(config['data_dir']+'/wwo_cache.csv')
        except ValueError:
            print("unable to retrieve any new weather data")
            hist_weather_df = hist_weather_df_cache

        joined_df = filtered_df.join(hist_weather_df, how='inner')
        joined_df.to_csv(config['data_dir']+'/prepped_data.csv')
        joined_df.to_pickle(config['data_dir']+'/prepped_data.pkl')

if __name__ == '__main__':
    retrieve_data()