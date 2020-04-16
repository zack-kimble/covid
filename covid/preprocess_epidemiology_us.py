#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:51:39 2020

@author: zack
"""

from pandas.tseries.offsets import DateOffset
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

#TODO: add onset to test taken time and onset to test confirmed time



config = dict(
    source_data = 'data/us_data_pivoted.pkl',
    index_cols = ['FIPS','date'],
    case_fatality_rate = .02,
    onset_to_death_mean=14,
    onset_to_death_sdev=5,
    incubation_time_mean=6,
    incubation_time_sdev=2,
    infection_case_rate=.65,
    incubation_log_normal = {'mean':1.621, 'stdev':.418},
    infectious_exponential = {'loc':5, 'scale':.666},
    onset_to_death_gamma = {'mean': 18.8, 'cov':.45}
)



#Incubation parameters: https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
#Infectious period parameters based on : https://www.medrxiv.org/content/10.1101/2020.03.05.20030502v1
#Percent of infections confirmed as cases (used post travel control number): https://science.sciencemag.org/content/early/2020/03/24/science.abb3221
#Onset to death timing https://www.thelancet.com/action/showPdf?pii=S1473-3099%2820%2930243-7 (weirdly given in mean and coefficient of variation of a gamma distribution)

def add_dates_to_index(df, offset, index_cols):
    # df.reset_index(inplace=True)
    keep_index = [col for col in index_cols if col != 'date']
    df.set_index(keep_index, inplace=True)
    offset_dates = df['date'].apply(DateOffset(days=offset)).rename('date')
    offset_dates = pd.MultiIndex.from_frame(offset_dates.reset_index())
    df = df.reset_index().set_index(index_cols)
    new_index = offset_dates.union(df.index)
    augmented_df = df.reindex(new_index)
    return augmented_df


def estimate_upstream_from_downstream(df, rate, offset_dist, index_cols, downstream_label, upstream_label):
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

    def _estimate_upstream_from_downstream(row, offsets_list, downstream_label, upstream_label, offset_dist):
        # row = row.reset_index()
        # copied_index = row[[col for col in index_cols if col != 'date']]
        non_date_index_fields = [col for col in index_cols if col != 'date']
        upstream_total = int(round(row[downstream_label] * 1 / rate))
        if upstream_total > 0:
            try:
                offsets = -1 * offset_dist.rvs(upstream_total)
            except TypeError:
                return
            offsets = pd.Series(np.round(offsets).astype(int))
            offsets = offsets.value_counts().rename(upstream_label).reset_index()
            offsets['date'] = offsets['index'].apply(lambda x: pd.Timedelta(days=x) + row['date'])
            for val in non_date_index_fields:
                offsets[val] = row[val]
            offsets.set_index(index_cols, inplace=True)
            offsets.drop(columns=['index'], inplace=True)
            offsets_list.append(offsets)

    _df.reset_index(inplace=True)
    _df.apply(_estimate_upstream_from_downstream, axis=1, offsets_list=offsets_list, downstream_label=downstream_label,
              upstream_label=upstream_label, offset_dist=offset_dist)
    offsets_df = pd.concat(offsets_list)
    offsets_df = offsets_df.groupby(offsets_df.index.names).sum()
    _df = df.join(offsets_df)
    _df[upstream_label].fillna(0, inplace=True)
    return _df


# def calc_r_proxy(cumulative_case_array, d):
#     """
#     Uses R proxy found in this paper:https://www.medrxiv.org/content/10.1101/2020.02.12.20022467v1.full.pdf
#
#     r(t,d) = (C(t+2d) - C(t + d) ) / ( C(t+d) - C(t)
#
#     Where d is the interval period in days, t is time index, and C(x) is cases at time x.
#
#     Is two weeks behind. :(
#     """
#     cumulative_case_array = np.array(cumulative_case_array)
#     c_t = cumulative_case_array[0]
#     c_t_d = cumulative_case_array[d]
#     c_t_2d = cumulative_case_array[2 * d]
#     r0 = (c_t_2d - c_t_d) / (c_t_d - c_t)
#     return r0

#
# def calc_rolling_r_proxy(cumulative_cases_series, d):
#     cumulative_cases_series = cumulative_cases_series[::-1]  # reverse because window function is only backwards
#     r_values = cumulative_cases_series.rolling(window=(2 * d) + 1).apply(calc_r_proxy, raw=True, kwargs=dict(d=d))
#     return r_values

def create_array_from_val(value, length):
    array = np.zeros(length)
    array[0] = value
    return array

#need this for making sure infections/incubations fall off correctly in tail.
def round_to_int_probablistic(num):
    return int(np.floor(num + np.random.random()))

def calculate_latent_and_infectious_population(df, new_infection_field, incubation_dist, infectious_dist, index_cols, prefix):

    latent_array = prefix + '_latent_array'
    infectious_array = prefix +'_infectious_array'
    prev_latent_array = 'prev_' + latent_array
    prev_infectious_array = 'prev_' + infectious_array
    total_latent = 'total_' + latent_array
    new_onsets = prefix + '_new_onsets'
    non_date_index_cols = [x for x in index_cols if x != 'date']

    _df = df.reset_index().set_index(index_cols).copy()
    _df[latent_array] = _df[new_infection_field].apply(create_array_from_val, length=14)
    _df[infectious_array] = list(np.zeros(shape=(len(_df), 14)))
    _df[new_onsets] = 0

    #create array of transition probalities
    incubation_probs = incubation_dist.sf(np.linspace(0, 13, 14))
    infection_probs = infectious_dist.sf(np.linspace(0, 13, 14))
    #fips_date_range = _df.groupby(['FIPS']).aggregate(start_date=('date', 'min'), end_date=('date', 'max'))
    start_date = _df.index.get_level_values('date').min() #TODO make date field name an arg
    end_date = _df.index.get_level_values('date').max()
    dates = pd.date_range(start_date, end_date)
    #loop through days and update all locations based on previous day
    for date in dates:
        #get previous day's date
        prev_date = date + pd.Timedelta(days=-1)
        prev_date_sub_df = _df.loc[pd.IndexSlice[:, (prev_date, date)], :]

        #Update Latent Array
        #First get previous day's latent array
        prev_latent_array_series = prev_date_sub_df[latent_array].groupby(level=non_date_index_cols).shift().loc[pd.IndexSlice[:, date]]
        #TODO: add date back to index here so I don't have to use .values later on when assigning back to _df
        #check for nan and replace with zero array if necessary. Must work on series of scalars, series of arrays, or mixed
        #pd.fillna won't fill accept arrays for fill values, so doing this the slower way only when necessary
        if np.any(pd.isna(prev_latent_array_series.sum(skipna=False))):
            prev_latent_array_series.loc[prev_latent_array_series.isna()] = list(np.zeros(shape = (len(prev_latent_array_series[prev_latent_array_series.isna()]),14)))

        #remember to not overwrite the latent_array[0] value written using create_array_from_val. Do ops on prev_laten_array, then add the two vectors.
        #calculate total latent at end of previous day to use for finding new onsets later
        prev_total_latent_series = prev_latent_array_series.apply(np.sum)
        #calculate how many move to next day or have onset and become infectious
        prev_latent_array_series = prev_latent_array_series.apply(lambda x: x * incubation_probs)
        prev_latent_array_series = prev_latent_array_series.apply(np.vectorize(round_to_int_probablistic)) #TODO maybe just go change function instead of using vectorize
        #Update the location within the infection trajectory (ie move day 1 infections to day 2)
        prev_latent_array_series = prev_latent_array_series.apply(np.roll, shift=1)
        #Put zero for the first day (will be replaced by existing value in latent_array)
        prev_latent_array_series.apply(lambda x: np.put(x, 0, 0)) #np.put edits in place, which seems ok. Also ends up operating on each value in each array when used directly with Series.apply()
        _df.loc[pd.IndexSlice[:, date], latent_array] = np.add(_df.loc[pd.IndexSlice[:, date],latent_array].values, prev_latent_array_series.values)
        #sum all latent infections for a day
        #_df.loc[pd.IndexSlice[:, date], total_latent] = _df.loc[pd.IndexSlice[:, date], latent_array].apply(np.sum)

        #Update Infectious Array
        #Calculate number of new onsets as total latent from previous day minus total latent of new day
        _df.loc[pd.IndexSlice[:, date], new_onsets] = prev_total_latent_series.values - prev_latent_array_series.apply(np.sum).values
        _df.loc[pd.IndexSlice[:, date], infectious_array] = _df.loc[pd.IndexSlice[:, date], new_onsets].apply(create_array_from_val, length=14)

        prev_infectious_array_series = prev_date_sub_df[infectious_array].groupby(level=non_date_index_cols).shift().loc[pd.IndexSlice[:, date]]
        #check for nan and replace with zero array if necessary. Must work on series of scalars, series of arrays, or mixed
        #pd.fillna won't fill accept arrays for fill values, so doing this the slower way only when necessary
        if np.any(pd.isna(prev_infectious_array_series.sum(skipna=False))):
            prev_infectious_array_series.loc[prev_infectious_array_series.isna()] = list(np.zeros(shape = (len(prev_infectious_array_series[prev_infectious_array_series.isna()]),14)))

        # remember to not overwrite the infectious_array[0] value written using create_array_from_val. Do ops on prev_laten_array, then add the two vectors.
        # calculate total infectious at end of previous day to use for finding new onsets later
        prev_total_infectious_series = prev_infectious_array_series.apply(np.sum)
        # calculate how many move to next day or have onset and become infectious
        prev_infectious_array_series = prev_infectious_array_series.apply(lambda x: x * infection_probs)
        prev_infectious_array_series = prev_infectious_array_series.apply(np.vectorize(
            round_to_int_probablistic))  # TODO maybe just go change function instead of using vectorize
        # Update the location within the infection trajectory (ie move day 1 infections to day 2)
        prev_infectious_array_series = prev_infectious_array_series.apply(np.roll, shift=1)
        # Put zero for the first day (will be replaced by existing value in infectious_array)
        prev_infectious_array_series.apply(lambda x: np.put(x, 0, 0))  # np.put edits in place, which seems ok. Also ends up operating on each value in each array when used directly with Series.apply()
        _df.loc[pd.IndexSlice[:, date], infectious_array] = np.add(_df.loc[pd.IndexSlice[:, date], infectious_array].values, prev_infectious_array_series.values)





        # #latent infections
        # #first get data from previous day
        # _df[prev_latent_array] = _df.groupby(level='FIPS')[latent_array].shift() #TODO parameterize 'FIPS"
        # #multiply times incubation prob to get number still incubating
        # _df[latent_array] = _df[prev_latent_array] * incubation_probs
        # #progress to next day of incubation
        # _df[latent_array] = _df[latent_array].apply(np.roll, shift=1)
        # #_df[latent_array] = _df[latent_array].apply(np.put, ind=0, v=0)
        #
        # #Round to integer stochastically
        # _df[latent_array] = _df[latent_array].apply(np.apply_along_axis, func1d=round_to_int_probablistic, axis=0)
        # #calculate number of new onsets (exiting incubation regardless of symptoms)
        # #_df[new_onsets] = _df[latent_array].apply()

        #_df[infectious_array] = _df.groupby(level='FIPS')[infectious_array].shift()


    _df[prefix + '_latent_population'] = _df[latent_array].apply(np.sum)
    _df[prefix + '_infectious_population'] = _df[infectious_array].apply(np.sum)
    return _df



cfr = config['case_fatality_rate']
onset_to_death_mean = config['onset_to_death_mean']
onset_to_death_sdev = config['onset_to_death_sdev']
infection_case_rate = config['infection_case_rate']
incubation_time_mean = config['incubation_log_normal']['mean']
incubation_time_sdev = config['incubation_log_normal']['stdev']
infectious_loc = config['infectious_exponential']['loc']
infectious_scale = config['infectious_exponential']['scale']
index_cols = config['index_cols']

#calculate infection_fatality_rate from cfr and infection to case rate
infection_fatality_rate = infection_case_rate * cfr

#define distribution of incubation period
incubation_dist = stats.lognorm(s=incubation_time_sdev,scale=np.exp(incubation_time_mean))

#define distribution of infectious period
infectious_dist = stats.expon(loc= infectious_loc, scale=infectious_scale)

#define distribution of onset to death
#calc shape and scale of gamma from mean, coefficient of variation
cov = config['onset_to_death_gamma']['cov']
mean = config['onset_to_death_gamma']['mean']
variance = (cov*mean)**2
scale = (variance/mean)**(1/3)
shape = mean/scale
onset_to_death_dist = stats.gamma(a=shape, scale=scale)

joined_df = pd.read_pickle(config['source_data'])
joined_df = joined_df.reset_index().set_index(index_cols)

#remove any values with non-unique indexes
unique_index = ~joined_df.index.duplicated(keep=False)
joined_df = joined_df.loc[unique_index]

groupby = [x for x in index_cols if x != 'date']
joined_df['new_deaths'] = joined_df.groupby(groupby)['deaths'].diff().clip(lower=0)
joined_df['new_cases'] = joined_df.groupby(groupby)['cases'].diff().clip(lower=0)
num_columns = joined_df.select_dtypes(include='number').columns
joined_df.loc[:,num_columns] = joined_df[num_columns].fillna(0)



# Estimate the number of cases in the past by looking at daily new deaths and propogating backwards
joined_df = estimate_upstream_from_downstream(joined_df, cfr, onset_to_death_dist, index_cols, 'new_deaths', 'cases_based_on_deaths')
# Estimate infections times from those cases
joined_df = estimate_upstream_from_downstream(joined_df, infection_case_rate, incubation_dist, index_cols, 'cases_based_on_deaths', 'infections_based_on_deaths')

# Shift confirmed cases backwards to transmission
joined_df = estimate_upstream_from_downstream(joined_df, infection_case_rate, incubation_dist, index_cols, 'new_cases', 'infections_based_on_cases')

joined_df.to_pickle('joined_df_test_us.pkl')

#joined_df = pd.read_pickle('joined_df_test_us.pkl')

#calculate infectious population

joined_df = calculate_latent_and_infectious_population(df=joined_df,
                                                    new_infection_field='infections_based_on_deaths',
                                                    incubation_dist=incubation_dist,
                                                    infectious_dist=infectious_dist,
                                                    index_cols=index_cols,
                                                    prefix='death_based')

joined_df = calculate_latent_and_infectious_population(df=joined_df,
                                                    new_infection_field='infections_based_on_cases',
                                                    incubation_dist=incubation_dist,
                                                    infectious_dist=infectious_dist,
                                                    index_cols=index_cols,
                                                    prefix='case_based')

joined_df.to_pickle('data/us_data_with_latent_populations.pkl')

