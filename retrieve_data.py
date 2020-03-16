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
import seaborn as sns
import numpy as np
from pandas.tseries.offsets import DateOffset
from scipy import stats
from wwo_hist import retrieve_hist_data

config = dict(
use_local = False,
index_col = ['Country/Region', 'Province/State', 'Lat', 'Long', 'date'],
weather_api_key = '',
cfr = .01,
onset_to_death_mean=-14,
onset_to_death_sdev=5,
incubation_time_mean=6,
incubation_time_sdev=2,
infection_to_case_rate=1
)

def get_melt_clean(url,value_name,use_local):   
    if use_local:
        df = pd.read_csv(url.split(sep='/')[-1])
    else:
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.text))
    
    id_cols = ['Country/Region','Province/State', 'Lat', 'Long']
    df = df.melt(id_vars=id_cols, var_name='date', value_name=value_name)
    df['date'] = pd.to_datetime(df.date)
    return df

def filter_countries(full_df, min_deaths, min_cases):
    include_countries = full_df[np.any((full_df.deaths >= min_deaths, full_df.cases >= min_cases),0)]['Country/Region'].unique()
    filtered_df = full_df[full_df['Country/Region'].apply(lambda x: x in include_countries)].copy()
    return filtered_df

def add_dates_to_index(df,offset,index_cols):
    #df.reset_index(inplace=True)
    keep_index = [col for col in index_cols if col != 'date']
    df.set_index(keep_index, inplace=True)
    offset_dates = df['date'].apply(DateOffset(days=offset)).rename('date')
    offset_dates = pd.MultiIndex.from_frame(offset_dates.reset_index())
    df = df.reset_index().set_index(index_cols)
    new_index = offset_dates.union(df.index)
    augmented_df = df.reindex(new_index)
    return augmented_df


# def estimate_cases_from_deaths(df, cfr, offset, index_cols):
#     """   

#     Parameters
#     ----------
#     df : pd.dataframe
#         DESCRIPTION.
#     cfr : float
#         DESCRIPTION.
#     offest : int
#         WHO China join mission says 2 to 8 weeks:https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf 

#     Returns
#     -------
#     df

#     """
#     df = add_dates_to_index(df, offset, index_cols)
#     cases_from_deaths = df['deaths'] * 1/cfr
#     cases_from_deaths = cases_from_deaths.rename('cases_based_on_deaths').reset_index()
#     cases_from_deaths['date'] = cases_from_deaths['date'].apply(DateOffset(days=offset))
#     cases_from_deaths.set_index(index_cols)
#     pd.concat([df,cases_from_deaths])

def estimate_upstream_from_downstream(df, rate, offset_mean, offset_std, index_cols, downstream_label, upstream_label):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    rate : INT
        The rate at which cases progress to the downstream data point. For estimating cases from deaths, this is CFR. For
        estimating cases from confirmed cases, this is the % of actual cases that get tested and confirmed.
    offset_mean : TYPE
        Mean time between the case onset and the observed data points in days
    offset_std : TYPE
        Standard deviation of time between the case onset and the observed data points in days
    index_cols : TYPE
        The columns in the DF that identify and individual data point.

    Returns
    -------
    Dataframe
        D.

    """
    _df = df.copy()
    offsets_list = []
    def _estimate_upstream_from_downstream(row, offsets_list, downstream_label, upstream_label):
        #row = row.reset_index()
        #copied_index = row[[col for col in index_cols if col != 'date']]
        non_date_index_fields = [col for col in index_cols if col != 'date']
        upstream_total = int(round(row[downstream_label] * 1/rate))
        if upstream_total > 0:
            try:
                offsets = stats.norm(loc=offset_mean, scale=offset_std).rvs(upstream_total) #TODO use something other than norm
            except TypeError:
                return
            offsets = pd.Series(np.round(offsets).astype(int))
            offsets = offsets.value_counts().rename(upstream_label).reset_index()
            offsets['date'] = offsets['index'].apply(lambda x: pd.Timedelta(days = x) + row['date'])
            for val in non_date_index_fields:
                offsets[val] = row[val]
            offsets.set_index(index_cols,inplace = True)
            offsets.drop(columns=['index'],inplace=True)
            offsets_list.append(offsets)
    _df.reset_index(inplace=True)
    _df.apply(_estimate_upstream_from_downstream, axis=1, offsets_list = offsets_list, downstream_label = downstream_label, upstream_label = upstream_label)
    offsets_df = pd.concat(offsets_list)
    offsets_df = offsets_df.groupby(offsets_df.index.names).sum()
    _df = df.join(offsets_df)
    _df[upstream_label].fillna(0,inplace=True)
    return _df

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
filtered_df = filter_countries(full_df,1,100)
filtered_df['new_deaths'] = filtered_df.groupby(['Country/Region','Province/State'])['deaths'].diff()
filtered_df['new_cases'] = filtered_df.groupby(['Country/Region','Province/State'])['cases'].diff()
filtered_df = add_dates_to_index(filtered_df, -14, join_cols)
filtered_df = filtered_df.select_dtypes(include='number').fillna(0)

cfr = config['cfr']
onset_to_death_mean = config['onset_to_death_mean']
onset_to_death_sdev = config['onset_to_death_sdev']
infection_to_case_rate = config['infection_to_case_rate']
incubation_time_mean = config['incubation_time_mean']
incubation_time_sdev = config['incubation_time_sdev']

# Estimate the number of cases in the past by looking at daily new deaths and propogating backwards
filtered_df = estimate_upstream_from_downstream(filtered_df, cfr, onset_to_death_mean, onset_to_death_sdev, join_cols, 'new_deaths', 'cases_based_on_deaths')
# Estimate infections times from those cases
filtered_df = estimate_upstream_from_downstream(filtered_df, infection_to_case_rate, incubation_time_mean, incubation_time_sdev, join_cols, 'cases_based_on_deaths', 'infections_based_on_deaths')
# Shift confirmed cases backwards to transmission (assumes perfect testing though, since there is little data on testing coverage)
filtered_df = estimate_upstream_from_downstream(filtered_df, infection_to_case_rate, incubation_time_mean, incubation_time_sdev, join_cols, 'new_cases', 'infections_based_on_cases')

# Make cumulative counts for the new fields
filtered_df['cumulative_cases_based_on_deaths'] = filtered_df.groupby(['Country/Region','Province/State'])['cases_based_on_deaths'].cumsum()
filtered_df['cumulative_infection_based_on_deaths'] = filtered_df.groupby(['Country/Region','Province/State'])['infections_based_on_deaths'].cumsum()
filtered_df['cumulative_infection_based_on_cases'] = filtered_df.groupby(['Country/Region','Province/State'])['infections_based_on_cases'].cumsum()

#calculate active cases
filtered_df['active_cases'] = filtered_df['cases'] - filtered_df['deaths'] - filtered_df['recoveries']
#TODO: Need a way to calculate active infections based on death. Have to calculate some number of non tested mild infections
#calculate percentage change
filtered_df['new_cases_as_percent_of_active'] = filtered_df['new_cases']/filtered_df['active_cases']
filtered_df['infections_based_on_deaths_as_percent_of_active'] = filtered_df['infections_based_on_deaths']/filtered_df['active_cases']
filtered_df['infections_based_on_cases_as_percent_of_active'] = filtered_df['infections_based_on_cases']/filtered_df['active_cases']



#chang index to lat/long and date for use with weather
filtered_df.reset_index(['Country/Region','Province/State'], inplace=True)

#load local weather cache and check of missing location/dates
try:
    hist_weather_df_cache = pd.read_csv('data/wwo_cache.csv')
    hist_weather_df_cache['date'] = pd.to_datetime(hist_weather_df_cache['date'])
    hist_weather_df_cache = hist_weather_df_cache.set_index(['Lat','Long','date'])

except FileNotFoundError:
    hist_weather_df_cache = pd.DataFrame()

#get unique missing lat/long date tuples
missing_indexes = filtered_df.index.difference(hist_weather_df_cache.index)
missing_val_df = filtered_df.loc[missing_indexes]
missing_val_agg = missing_val_df.reset_index().groupby(['Lat','Long']).aggregate(start_date=('date','min'),end_date=('date','max')).reset_index()
missing_val_agg['lat_long_string'] = [str(x) + ','+ str(y) for x, y in zip(missing_val_agg['Lat'], missing_val_agg['Long'])]

# lat_long = filtered_df.reset_index()[['Lat','Long']].drop_duplicates()
# lat_long_strings =[str(x) + ','+ str(y) for x, y in zip(lat_long['Lat'], lat_long['Long'])]

# #get start and end dates 
# start_date = filtered_df.index.get_level_values('date').min()
# end_date = filtered_df.index.get_level_values('date').max()

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
try:    
    hist_weather_df_new = pd.concat(hist_weather_list)
    hist_weather_df_new['Lat'] = hist_weather_df_new['location'].apply(lambda x: x.split(',')[0])
    hist_weather_df_new['Long'] = hist_weather_df_new['location'].apply(lambda x: x.split(',')[1])
    hist_weather_df_new.rename(columns={'date_time':'date'})
    hist_weather_df_new.set_index(['Lat','Long','date'],inplace=True)
    hist_weather_df = pd.concat([hist_weather_df_cache, hist_weather_df_new],verify_integrity=True)

except ValueError:
    

hist_weather_df.to_csv('data/wwo_cache.csv')

joined_df = filtered_df.join(hist_weather_df, how='inner')



#noaa_data = pd.read_csv('data/2020.csv')

#filtered_df['death_onset_date']
# filtered_df.select_dtypes(include='number').columns
# filtered_df['cases'].fillna(0)
#
# countries_with_10_deaths = deaths_df[np.any(full_df.deaths >= 10,full_df.cases]['Country/Region'].unique()
# deaths_df = deaths_df[deaths_df['Country/Region'].apply(lambda x: x in countries_with_deaths)]
# deaths_df = deaths_df[deaths_df['Country/Region'].apply(lambda x: x in countries_with_10_deaths)]
# deaths_df_by_country = deaths_df.groupby(['date','Country/Region']).deaths.sum().reset_index()
#
# sns.set(rc={'figure.figsize':(20,8.27)})
# sns.lineplot(data = deaths_df_by_country, x = 'date', y = 'deaths', hue='Country/Region')


# filtered_df.reset_index()['date'][20] + pd.Timedelta()

# df, cfr, offset_mean, offset_std, index_cols = filtered_df, .01, -14, 5, join_cols
# row = filtered_df[filtered_df["index"]==21877]
# filtered_df.loc[row.index]
# filtered_df[row[index_cols]]

# idx = pd.IndexSlice
# df.loc[idx[:,['United Kingdom'],:,:,:]]
# df.loc[slice(None),slice('United Kingdom'),slice(None),slice(None),slice(None)]
# example = df.loc['United Kingdom','United Kingdom']
# dir(df.index)