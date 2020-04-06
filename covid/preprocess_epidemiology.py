from pandas.tseries.offsets import DateOffset
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

#calc shape and scale of gamma from mean, coefficient of variation




config = dict(
    index_cols = ['Country/Region', 'Province/State', 'Lat', 'Long', 'date'],              
              cfr = .01,
onset_to_death_mean=14,
onset_to_death_sdev=5,
incubation_time_mean=6,
incubation_time_sdev=2,
infection_to_case_rate=.65,
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


def calc_r_proxy(cumulative_case_array, d):
    """
    Uses R proxy found in this paper:https://www.medrxiv.org/content/10.1101/2020.02.12.20022467v1.full.pdf

    r(t,d) = (C(t+2d) - C(t + d) ) / ( C(t+d) - C(t)

    Where d is the interval period in days, t is time index, and C(x) is cases at time x.

    Is two weeks behind. :(
    """
    cumulative_case_array = np.array(cumulative_case_array)
    c_t = cumulative_case_array[0]
    c_t_d = cumulative_case_array[d]
    c_t_2d = cumulative_case_array[2 * d]
    r0 = (c_t_2d - c_t_d) / (c_t_d - c_t)
    return r0


def calc_rolling_r_proxy(cumulative_cases_series, d):
    cumulative_cases_series = cumulative_cases_series[::-1]  # reverse because window function is only backwards
    r_values = cumulative_cases_series.rolling(window=(2 * d) + 1).apply(calc_r_proxy, raw=True, kwargs=dict(d=d))
    return r_values

def create_array_from_val(value, length):
    array = np.zeros(length)
    array[0] = value
    return array  

#need this for making sure infections/incubations fall off correctly in tail.
def round_to_int_probablistic(num):
    return int(np.floor(num + np.random.random()))

def calculate_latent_and_infectious_population(df, new_infection_field, incubation_dist, infectious_dist, index_cols,prefix):
    _df = df.reset_index().set_index(index_cols).copy()
    _df[prefix+'_latent_array'] = _df[new_infection_field].apply(create_array_from_val, length=14)
    _df[prefix+'_infectious_array'] = list(np.zeros(shape=(len(_df),14)))
    # loop through rows and update each date base on the previous date (within location)
    for key in _df.index.values:
        row = _df.loc[[key]].copy()
        #Try to get previous day data
        prev_key = np.array(key)
        prev_key[-1] = prev_key[-1] + pd.Timedelta(days=-1)
        prev_key = tuple(prev_key)
        try:
            prev_row = _df.loc[[prev_key]]
            prev_latent = prev_row[prefix+'_latent_array'].values[0]
            prev_infectious = prev_row[prefix+'_infectious_array'].values[0]
        except KeyError:
            prev_row = None
            prev_latent = np.zeros(14)
            prev_infectious = np.zeros(14)
        new_latent = row[prefix+'_latent_array'].values[0]
        new_infectious = row[prefix+'_infectious_array'].values[0]
        #People with incubating virus either continue in incubation or proceed to infectious state
        if np.any(prev_latent):
            for day_number in range(1, len(prev_latent)-1):
                prev_day_num = day_number - 1
                incubation_prob = incubation_dist.sf(day_number)/incubation_dist.sf(prev_day_num)
                new_latent[day_number] = round_to_int_probablistic(prev_latent[prev_day_num] * incubation_prob)
                new_infectious[0] += prev_latent[day_number-1] - new_latent[day_number]
            _df.loc[[key]][prefix+'_latent_array'] = list(new_latent.reshape(1, 14))
        #Infectious individuals either continue being infectious or stop being infectious
        if np.any(prev_infectious):
            for day_number in range(1, len(prev_infectious)-1):
                prev_day_num = day_number - 1
                infectious_prob = infectious_dist.sf(day_number)/infectious_dist.sf(prev_day_num)
                new_infectious[day_number] = round_to_int_probablistic(prev_infectious[prev_day_num] * infectious_prob)
            #remove positive cases from infectious pool (assume quarantined in some way)
            n_infectious = sum(prev_infectious)
            if n_infectious == 0:
                p_caught = 0
            else:
                p_caught = row['new_cases'].values/n_infectious
            new_infectious = new_infectious * (1-p_caught)
            _df.loc[[key]][prefix+'_infectious_array'] = list(new_infectious.reshape(1,14))
    _df[prefix + '_latent_population'] = _df[prefix+'_latent_array'].apply(np.sum)
    _df[prefix + '_infectious_population'] = _df[prefix+'_infectious_array'].apply(np.sum)
    return _df
    
            

cfr = config['cfr']
onset_to_death_mean = config['onset_to_death_mean']
onset_to_death_sdev = config['onset_to_death_sdev']
infection_to_case_rate = config['infection_to_case_rate']
incubation_time_mean = config['incubation_log_normal']['mean']
incubation_time_sdev = config['incubation_log_normal']['stdev']
infectious_loc = config['infectious_exponential']['loc']
infectious_scale = config['infectious_exponential']['scale']
index_cols = config['index_cols']

#define distribution of incubation period
incubation_dist = stats.lognorm(s=incubation_time_sdev,scale=np.exp(incubation_time_mean))

#define distribution of infectious period
infectious_dist = stats.expon(loc= infectious_loc, scale=infectious_scale)

#define distribution of onset to death
#First convert to shape and scale
cov = config['onset_to_death_gamma']['cov']
mean = config['onset_to_death_gamma']['mean']
variance = (cov*mean)**2
scale = (variance/mean)**(1/3)
shape = mean/scale
onset_to_death_dist = stats.gamma(a=shape, scale=scale)

#joined_df = pd.read_csv('data/prepped_data.csv')
joined_df = pd.read_pickle('data/prepped_data.pkl')

joined_df = joined_df.reset_index().set_index(index_cols)
# Estimate the number of cases in the past by looking at daily new deaths and propogating backwards
joined_df = estimate_upstream_from_downstream(joined_df, cfr, onset_to_death_dist, index_cols, 'new_deaths', 'cases_based_on_deaths')
# Estimate infections times from those cases
joined_df = estimate_upstream_from_downstream(joined_df, infection_to_case_rate, incubation_dist, index_cols, 'cases_based_on_deaths', 'infections_based_on_deaths')
# Shift confirmed cases backwards to transmission 
joined_df = estimate_upstream_from_downstream(joined_df, infection_to_case_rate, incubation_dist, index_cols, 'new_cases', 'infections_based_on_cases')

joined_df.to_pickle('joined_df_test.pkl')

#joined_df = pd.read_pickle('joined_df_test.pkl')

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

joined_df.to_pickle('data/df_with_arrays_4_3.pkl')


# #joined_df = pd.read_pickle('data/df_with_arrays_4_3.pkl')
# #Get previous day's numbers for latent and infectious population
# joined_df['prev_total_latent'] = joined_df.groupby(['Country/Region','Province/State'])['total_latent'].shift()
# joined_df['prev_total_infectious'] = joined_df.groupby(['Country/Region','Province/State'])['total_infectious'].shift()
#
# # Make cumulative counts for the new fields
# joined_df['cumulative_cases_based_on_deaths'] = joined_df.groupby(['Country/Region','Province/State'])['cases_based_on_deaths'].cumsum()
# joined_df['cumulative_infection_based_on_deaths'] = joined_df.groupby(['Country/Region','Province/State'])['infections_based_on_deaths'].cumsum()
# joined_df['cumulative_infection_based_on_cases'] = joined_df.groupby(['Country/Region','Province/State'])['infections_based_on_cases'].cumsum()
#
#
# #TODO: Need a way to calculate active infections based on death. Have to calculate some number of non tested mild infections
# #calculate percentage change
# joined_df['new_cases_as_percent_of_active'] = joined_df['new_cases']/joined_df['active_cases']
# joined_df['infections_based_on_deaths_as_percent_of_active'] = joined_df['infections_based_on_deaths']/joined_df['active_cases']
# joined_df['infections_based_on_cases_as_percent_of_active'] = joined_df['infections_based_on_cases']/joined_df['active_cases']
#
# joined_df['new_cases_as_percent_of_infectious'] = joined_df['new_cases']/joined_df['total_infectious']
# joined_df['infections_based_on_deaths_as_percent_of_infectious'] = joined_df['infections_based_on_deaths']/joined_df['total_infectious']
# joined_df['infections_based_on_cases_as_percent_of_infectious'] = joined_df['infections_based_on_cases']/joined_df['total_infectious']
#
# sns.set(rc={'figure.figsize': (20, 8.27)})
#
# sns.violinplot(data=joined_df[joined_df['new_cases_as_percent_of_active'].notna()],x='maxtempC', y='new_cases_as_percent_of_active')
# sns.boxplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_cases_as_percent_of_active')
# sns.boxenplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_cases_as_percent_of_active', outlier_prop=.1)
#
# filter = np.all(np.array([
#      joined_df['infections_based_on_deaths_as_percent_of_infectious'].notna().values,
#      joined_df['infections_based_on_deaths_as_percent_of_infectious']<10,
#      joined_df['total_infectious']>1000,
#     ]),axis=0)
# data = joined_df.reset_index()[filter]
#
#
# sns.boxplot(
#     data=data,
#     x='maxtempC',
#     y='infections_based_on_deaths_as_percent_of_infectious')
#
# sns.scatterplot(
#     data=data,
#     x='maxtempC',
#     y='infections_based_on_deaths_as_percent_of_infectious',
#     hue="Country/Region")
#
# sns.scatterplot(
#     data=data,
#     x='maxtempC',
#     y='infections_based_on_deaths_as_percent_of_infectious',
#     hue="total_infectious")
#
# line_df = data[data['Country/Region']=='US'].set_index('date')[['new_cases','infections_based_on_deaths','infections_based_on_cases']]
# sns.lineplot(
#     data=line_df
#
#     )
#
# box_df = joined_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['new_cases_as_percent_of_active'],how="all")

