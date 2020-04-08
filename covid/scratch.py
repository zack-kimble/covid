#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:36:59 2020

@author: zack
"""

import seaborn as sns
import pandas as pd
from retrieve_data import calc_r_0, calc_rolling_r0

joined_df = pd.read_csv('data/prepped_data.csv')

joined_df['r'] = calc_rolling_r(joined_df.cases, 5)

test = joined_df.set_index(['Country/Region','Province/State', 'date'])[['cases','r0']]

sns.set(rc={'figure.figsize':(20,8.27)})
sns.violinplot(data=joined_df[joined_df['new_cases_as_percent_of_active'].notna()],x='maxtempC', y='new_cases_as_percent_of_active')
sns.boxplot(data=joined_df[joined_df['new_cases_as_percent_of_active'].notna()],x='maxtempC', y='new_cases_as_percent_of_active')
sns.boxplot(data=joined_df[joined_df['infections_based_on_deaths_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_deaths_as_percent_of_active')
sns.boxplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_cases_as_percent_of_active')
sns.boxenplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_cases_as_percent_of_active', outlier_prop=.1)

joined_df[joined_df['Country/Region'] == 'Italy']

joined_df['Country/Region'].unique()

joined_df.groupby(['maxtempC'])['new_cases'].sum()
joined_df.groupby(['date'])['cases'].sum()
cases_df.groupby(['date'])['cases'].sum()
full_df.groupby(['date'])['cases'].sum()
filtered_df.groupby(['date'])['cases'].sum()
x = joined_df[joined_df['maxtempC']==34]
joined_df

thailand = joined_df[joined_df['Country/Region']=='Thailand']


class infectious_period_dist(stats.lognorm):
    def sf(x):
        if x < 5:
            return 0
        else:
            return super.sf(x)
        
        
joined_df = pd.read_pickle('data/df_with_arrays_3_31.pkl')

view = joined_df[['infections_based_on_deaths','latent_array','infectious_array']]

test_df = joined_df.iloc[0:1500].copy()

argentina = view.loc[['Argentina']]

joined_df['total_latent'] = joined_df['latent_array'].apply(np.sum)
joined_df['total_infectious'] = joined_df['infectious_array'].apply(np.sum)
joined_df['prev_total_latent'] = joined_df.groupby(joined_df.index.names)['total_latent'].rolling(window=1).apply(sum).reset_index()

view = joined_df[['infections_based_on_deaths','latent_array','total_latent','prev_total_latent','cases']]


x = test_df['total_latent'].rolling(window=DateOffset(days=-1) ,closed='right').apply(sum)
test_df['prev_total_latent'] = test_df.groupby(['Country/Region','Province/State'])['total_latent'].shift()
x = test_df.groupby(['Country/Region','Province/State'])['total_latent'].shift()


new_infection_field = 'infections_based_on_deaths'
df = joined_df.copy()[0:2]
rows = [row for row in _df.reset_index().itertuples()]
row = rows[0]
type(row)
row['_1'
_df.loc[[_df.index.values[1]]]
x = _df['infected_array']
key =_df.index.values[0]

s=config['incubation_log_normal']['stdev']
loc=0
scale = np.exp(config['incubation_log_normal']['mean'])

x = np.linspace(stats.lognorm.ppf(0.01, s,scale=scale),stats.lognorm.ppf(0.99, s, scale=scale), 100)

sns.lineplot(x=x,y=stats.lognorm.pdf(x,s,scale=scale))
sns.lineplot(x=x,y=stats.lognorm.cdf(x,s,scale=scale))
sns.lineplot(x=x,y=stats.lognorm.sf(x,s,scale=scale))
sns.lineplot(x=x,y=stats.lognorm.sf(x,s,scale=scale)/stats.lognorm.sf(x-1,s,scale=scale))

stats.lognorm.sf(5,s,scale=scale)

sns.lineplot(x=x,y=stats.expon.sf(x,loc=5,scale=.75))

x = np.linspace(stats.gamma.ppf(0.01, shape,scale=scale),stats.gamma.ppf(0.99, shape, scale=scale), 100)
sns.lineplot(x=x,y=stats.gamma.pdf(x,shape,loc=scale))

hist_weather_df_cache = pd.read_csv('data/wwo_cache.csv')
hist_weather_df_cache['date'] = pd.to_datetime(hist_weather_df_cache['date'])
hist_weather_df_cache = hist_weather_df_cache.set_index(['Lat', 'Long', 'date'])
hist_weather_df_cache.loc[hist_weather_df_cache.index.drop_duplicates(keep=False)].to_csv('data/wwo_cache.csv')
hist_weather_df_cache = hist_weather_df_cache.loc[hist_weather_df_cache.index.drop_duplicates(keep=False)]
hist_weather_df_cache.index.is_unique

stats.gamma(a=shape, scale=scale)


#safegraph data exploration
import pandas as pd
social_distance = pd.read_csv('/home/zack/safegraph_data/social-distancing/v1/2020/02/01/2020-02-01-social-distancing.csv.gz')
weekly_patter = pd.read_csv('/home/zack/safegraph_data/weekly-patterns/v1/main-file/2020-03-22-weekly-patterns.csv.gz')


full_df.loc[[None]]