import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

#load social distance and prep for join
social_dist_county_df = pd.read_pickle('data/social_dist_county_df.pkl').reset_index()
social_dist_county_df['date'] = pd.to_datetime(social_dist_county_df['date_range_start'], utc=True).dt.date
social_dist_county_df = social_dist_county_df.set_index(['county_fips', 'date'])
social_dist_county_df['median_home_dwell_time_pct'] = social_dist_county_df['weighted_median_home_dwell_time'] / (24*60)


#load epidemiology and prep for join
epidemiology_df = pd.read_pickle('data/us_data_with_latent_populations.pkl').reset_index()
epidemiology_df['county_fips'] = epidemiology_df['UID'].apply(lambda x: x[3:])
epidemiology_df = epidemiology_df.set_index(['county_fips', 'date'])
epidemiology_df['infections_as_ratio_of_case_based_infectious_population'] = epidemiology_df['infections_based_on_cases'] / epidemiology_df[
    'case_based_infectious_population']

#check index overlap
epidemiology_df.index.difference(social_dist_county_df.index)
 


sd_epi_df = epidemiology_df.join(social_dist_county_df, how='inner')

#sum(sd_epi_df['pct_staying_home'].isna())
#sum(sd_epi_df['infections_as_ratio_of_case_based_infectious_population'].isna())

with pd.option_context('mode.use_inf_as_na', True):
    #sd_epi_df = sd_epi_df.dropna(subset=['infections_as_ratio_of_case_based_infectious_population'])
       
    data_filter = np.all(np.array([
            sd_epi_df['infections_as_ratio_of_case_based_infectious_population'].notna().values,
            sd_epi_df['infections_as_ratio_of_case_based_infectious_population'] < 2,
            sd_epi_df['case_based_infectious_population'] > 20,
            sd_epi_df.reset_index()['date'] < pd.datetime.today() - pd.Timedelta(10, 'days')
    ]), axis=0)

data = sd_epi_df[data_filter].copy()

sns.kdeplot(data=data['pct_staying_home'], data2=data['infections_as_ratio_of_case_based_infectious_population'])
sns.regplot(data=data, x='pct_staying_home', y='infections_as_ratio_of_case_based_infectious_population')

X = data['pct_staying_home'].values.reshape(-1,1)
y = data['infections_as_ratio_of_case_based_infectious_population']
reg = LinearRegression().fit(X,y)

print(reg.coef_)

#Look at dwell time instead
sns.kdeplot(data=data['median_home_dwell_time_pct'], data2=data['infections_as_ratio_of_case_based_infectious_population'])
sns.regplot(data=data, x='median_home_dwell_time_pct', y='infections_as_ratio_of_case_based_infectious_population')

X = data['median_home_dwell_time_pct'].values.reshape(-1,1)
y = data['infections_as_ratio_of_case_based_infectious_population']
reg = LinearRegression().fit(X,y)

print(reg.coef_)


#Looking over time at aggregate level
data['n_home'] = data['pct_staying_home'] * data['population']
us_aggregate = data.groupby('date').aggregate(
    infections_based_on_cases = ("infections_based_on_cases", sum),
    case_based_infectious_population = ('case_based_infectious_population', sum),
    population = ('population',sum),
    n_home = ('n_home',sum)
    ).reset_index()

us_aggregate['infections_as_ratio_of_case_based_infectious_population'] = us_aggregate['infections_based_on_cases'] / us_aggregate[
    'case_based_infectious_population']

us_aggregate['pct_staying_home'] = us_aggregate['n_home']/us_aggregate['population']

fig, ax = plt.subplots()

ax = sns.lineplot(data=us_aggregate, x='date', y='infections_as_ratio_of_case_based_infectious_population', ax=ax)
ax2 = sns.lineplot(data=us_aggregate, x='date', y='pct_staying_home')

locator = mdates.AutoDateLocator(minticks=12, maxticks=18)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

fig.show()

sns.set(rc={'figure.figsize': (20, 8.27)})

#Looking over time at top 200 counties
top_50 = sd_epi_df['new_cases'].groupby('county_fips').sum().sort_values()[-50:].index
with pd.option_context('mode.use_inf_as_na', True):
    #sd_epi_df = sd_epi_df.dropna(subset=['infections_as_ratio_of_case_based_infectious_population'])
       
    data_filter = np.all(np.array([
            sd_epi_df['infections_as_ratio_of_case_based_infectious_population'].notna().values,
            sd_epi_df.reset_index()['date'] < pd.datetime.today() - pd.Timedelta(10, 'days')
    ]), axis=0)

data = sd_epi_df.loc[top_50].reset_index()
sns.relplot(data=data, x='date', y='infections_as_ratio_of_case_based_infectious_population',col='Combined_Key',kind='line', col_wrap=5)
sns.relplot(data=data, x='date', y='pct_staying_home',col='Combined_Key',kind='line', col_wrap=5)
locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
