#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:10:01 2020

@author: zack
"""

import requests
import pandas as pd
import io
import numpy as np

config = dict(
    use_local=False,
    index_col=['UID','date'],
    desc_col = ['iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key'],
    data_dir='data/'
)


def get_df(url, use_local, data_dir):
    file_name = url.split(sep='/')[-1]
    if use_local:
        df = pd.read_csv(data_dir + file_name)
    else:
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv(data_dir + '/' + file_name, index=False)
    return df

def melt_df(df, pivot_cols, value_name):
    df = df.melt(id_vars=pivot_cols, var_name='date', value_name=value_name)
    df['date'] = pd.to_datetime(df.date)
    return df

pivot_cols = [x for x in config['index_col'] if x != 'date'] + config['desc_col']

deaths_df = get_df(
    url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv',
    use_local=config['use_local'],
    data_dir=config['data_dir'],
    )
deaths_df = melt_df(deaths_df, value_name = 'deaths',     pivot_cols = pivot_cols + ['Population'])

cases_df = get_df(
    url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv',
    use_local=config['use_local'],
    data_dir=config['data_dir'],
    )
cases_df = melt_df(cases_df, value_name = 'cases',     pivot_cols = pivot_cols)

lookup_df = get_df(
    url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv',
    use_local=config['use_local'],
    data_dir=config['data_dir'],
    )

    
join_cols = config['index_col']
full_df = pd.merge(deaths_df,cases_df[join_cols+['cases']], on=join_cols)
full_df = full_df.dropna(subset=join_cols)
full_df['UID'] = full_df['UID'].apply(lambda x: str(int(x)))

#full_df = full_df.set_index(join_cols).select_dtypes(include='number').fillna(0)

full_df.to_csv(config['data_dir']+'us_data_pivoted.csv')
full_df.to_pickle(config['data_dir']+'us_data_pivoted.pkl')

lookup_df.to_csv(config['data_dir']+'jhu_location_lookup.csv')