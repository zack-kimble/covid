#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 22:11:16 2020

@author: zack
"""

import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

joined_df = pd.read_pickle('data/us_data_with_latent_populations.pkl')
joined_df.columns



joined_df['infections_as_ratio_of_case_based_infectious_population'] = joined_df['infections_based_on_cases'] / joined_df[
    'case_based_infectious_population']

us_aggregate = joined_df.groupby('date').sum().reset_index()
us_aggregate['infections_as_ratio_of_case_based_infectious_population'] = us_aggregate['infections_based_on_cases'] / us_aggregate[
    'case_based_infectious_population']



filter = np.all(np.array([
    joined_df['infections_as_percent_of_case_based_infectious_population'].notna().values,
    joined_df['infections_as_percent_of_case_based_infectious_population'] < 10,
    joined_df['case_based_infectious_population'] > 100,
]), axis=0)
data = joined_df.reset_index()[filter]


ax2 = plt.twinx()

sns.relplot(data=data, x='date', y='infections_as_percent_of_case_based_infectious_population',col='FIPS',kind='line', col_wrap=20)

sns.set(rc={'figure.figsize': (20, 8.27)})
import matplotlib.ticker as ticker



fig, ax = plt.subplots()

ax = sns.lineplot(data=us_aggregate, x='date', y='infections_as_ratio_of_case_based_infectious_population', ax=ax)

locator = mdates.AutoDateLocator(minticks=12, maxticks=18)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)


fig.show()

plt.xticks( range(12),  rotation=17 )

plt.gcf()
plt.show()
ax.plot()


joined_df.index.get_level_values('FIPS').unique()

len(data.FIPS.unique())