from pandas.tseries.offsets import DateOffset
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


config = dict(
    index_cols = ['Country/Region', 'Province/State', 'Lat', 'Long', 'date'],              
              cfr = .01,
onset_to_death_mean=-14,
onset_to_death_sdev=5,
#incubation_time_mean=-6,
#incubation_time_sdev=2,
infection_to_case_rate=.65,
incubation_log_normal = {'mean':1.621, 'stdev':.418},
infectious_exponential = {'loc':5, 'scale':.666})

joined_df = pd.read_pickle('data/df_with_arrays_4_3.pkl')
# Get previous day's numbers for latent and infectious population
joined_df['prev_case_based_latent_population'] = joined_df.groupby(['Country/Region', 'Province/State'])['case_based_latent_population'].shift()
joined_df['prev_case_based_infectious_population'] = joined_df.groupby(['Country/Region', 'Province/State'])['case_based_infectious_population'].shift()

# Make cumulative counts for the new fields
joined_df['cumulative_cases_based_on_deaths'] = joined_df.groupby(['Country/Region', 'Province/State'])[
    'cases_based_on_deaths'].cumsum()
joined_df['cumulative_infection_based_on_deaths'] = joined_df.groupby(['Country/Region', 'Province/State'])[
    'infections_based_on_deaths'].cumsum()
joined_df['cumulative_infection_based_on_cases'] = joined_df.groupby(['Country/Region', 'Province/State'])[
    'infections_based_on_cases'].cumsum()

# TODO: Need a way to calculate active infections based on death. Have to calculate some number of non tested mild infections
# calculate percentage change
joined_df['new_cases_as_percent_of_active'] = joined_df['new_cases'] / joined_df['active_cases']
# joined_df['infections_based_on_deaths_as_percent_of_active'] = joined_df['infections_based_on_deaths'] / joined_df[
#     'active_cases']
joined_df['infections_based_on_cases_as_percent_of_active'] = joined_df['infections_based_on_cases'] / joined_df[
    'active_cases']

joined_df['new_cases_as_percent_of_infectious'] = joined_df['new_cases'] / joined_df['case_based_infectious_population']
# joined_df['infections_based_on_deaths_as_percent_of_infectious'] = joined_df['infections_based_on_deaths'] / joined_df[
#     'case_based_infectious_population']
joined_df['infections_based_on_cases_as_percent_of_infectious'] = joined_df['infections_based_on_cases'] / joined_df[
    'case_based_infectious_population']

sns.set(rc={'figure.figsize': (20, 8.27)})

# sns.violinplot(data=joined_df[joined_df['new_cases_as_percent_of_active'].notna()], x='maxtempC',
#                y='new_cases_as_percent_of_active')
# sns.boxplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()], x='maxtempC',
#             y='infections_based_on_cases_as_percent_of_active')
# sns.boxenplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()], x='maxtempC',
#               y='infections_based_on_cases_as_percent_of_active', outlier_prop=.1)

top_10_by_cases = joined_df.groupby('Country/Region')['cases'].max().sort_values()[-10:].index.values

joined_df = joined_df.loc[top_10_by_cases]

filter = np.all(np.array([
    joined_df['infections_based_on_cases_as_percent_of_infectious'].notna().values,
    #joined_df['infections_based_on_cases_as_percent_of_infectious'] < 10,
    #joined_df['case_based_infectious_population'] > 100,
]), axis=0)
data = joined_df.reset_index()[filter]



sns.boxplot(
    data=data,
    x='maxtempC',
    y='infections_based_on_deaths_as_percent_of_infectious')

sns.scatterplot(
    data=data,
    x='maxtempC',
    y='infections_based_on_deaths_as_percent_of_infectious',
    hue="Country/Region")

sns.scatterplot(
    data=data,
    x='maxtempC',
    y='infections_based_on_deaths_as_percent_of_infectious',
    hue="case_based_infectious_population")

line_df = data[data['Country/Region'] == 'France'].set_index('date')[
    ['new_cases',  'infections_based_on_cases','case_based_infectious_population']]

sns.lineplot(data=line_df)


ax2 = plt.twinx()
sns.lineplot(data=data[data['Country/Region']=='France'], x='date', y='infections_based_on_cases_as_percent_of_infectious', color="y", ax=ax2)

sns.lineplot(data=data, x='date', y='infections_based_on_cases_as_percent_of_infectious',hue='Country/Region')
sns.relplot(data=data, x='date', y='infections_based_on_cases_as_percent_of_infectious',row='Country/Region', col='Province/State',kind='line')



box_df = joined_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['new_cases_as_percent_of_active'], how="all")