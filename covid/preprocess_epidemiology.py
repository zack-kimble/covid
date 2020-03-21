from pandas.tseries.offsets import DateOffset
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np

config = dict(
    index_cols = ['Country/Region', 'Province/State', 'Lat', 'Long', 'date'],              
              cfr = .01,
onset_to_death_mean=-14,
onset_to_death_sdev=5,
incubation_time_mean=6,
incubation_time_sdev=2,
infection_to_case_rate=1)

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
        # row = row.reset_index()
        # copied_index = row[[col for col in index_cols if col != 'date']]
        non_date_index_fields = [col for col in index_cols if col != 'date']
        upstream_total = int(round(row[downstream_label] * 1 / rate))
        if upstream_total > 0:
            try:
                offsets = stats.norm(loc=offset_mean, scale=offset_std).rvs(
                    upstream_total)  # TODO change to weibull and get params from papers
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
              upstream_label=upstream_label)
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


def calculate_infectious_population():
    pass

cfr = config['cfr']
onset_to_death_mean = config['onset_to_death_mean']
onset_to_death_sdev = config['onset_to_death_sdev']
infection_to_case_rate = config['infection_to_case_rate']
incubation_time_mean = config['incubation_time_mean']
incubation_time_sdev = config['incubation_time_sdev']
index_cols = config['index_cols']

#joined_df = pd.read_csv('data/prepped_data.csv')
joined_df = pd.read_pickle('data/prepped_data.pkl')

# Estimate the number of cases in the past by looking at daily new deaths and propogating backwards
joined_df = estimate_upstream_from_downstream(joined_df, cfr, onset_to_death_mean, onset_to_death_sdev, index_cols, 'new_deaths', 'cases_based_on_deaths')
# Estimate infections times from those cases
joined_df = estimate_upstream_from_downstream(joined_df, infection_to_case_rate, incubation_time_mean, incubation_time_sdev, index_cols, 'cases_based_on_deaths', 'infections_based_on_deaths')
# Shift confirmed cases backwards to transmission (assumes perfect testing though, since there is little data on testing coverage)
joined_df = estimate_upstream_from_downstream(joined_df, infection_to_case_rate, incubation_time_mean, incubation_time_sdev, index_cols, 'new_cases', 'infections_based_on_cases')

# Make cumulative counts for the new fields
joined_df['cumulative_cases_based_on_deaths'] = joined_df.groupby(['Country/Region','Province/State'])['cases_based_on_deaths'].cumsum()
joined_df['cumulative_infection_based_on_deaths'] = joined_df.groupby(['Country/Region','Province/State'])['infections_based_on_deaths'].cumsum()
joined_df['cumulative_infection_based_on_cases'] = joined_df.groupby(['Country/Region','Province/State'])['infections_based_on_cases'].cumsum()


#TODO: Need a way to calculate active infections based on death. Have to calculate some number of non tested mild infections
#calculate percentage change
joined_df['new_cases_as_percent_of_active'] = joined_df['new_cases']/joined_df['active_cases']
joined_df['infections_based_on_deaths_as_percent_of_active'] = joined_df['infections_based_on_deaths']/joined_df['active_cases']
joined_df['infections_based_on_cases_as_percent_of_active'] = joined_df['infections_based_on_cases']/joined_df['active_cases']


sns.set(rc={'figure.figsize': (20, 8.27)})

sns.violinplot(data=joined_df[joined_df['new_cases_as_percent_of_active'].notna()],x='maxtempC', y='new_cases_as_percent_of_active')
sns.boxplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_cases_as_percent_of_active')
sns.boxenplot(data=joined_df[joined_df['infections_based_on_cases_as_percent_of_active'].notna()],x='maxtempC', y='infections_based_on_cases_as_percent_of_active', outlier_prop=.1)


box_df = joined_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['new_cases_as_percent_of_active'],how="all")