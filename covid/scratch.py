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