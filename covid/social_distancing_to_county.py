import os, glob
import pandas as pd
import numpy as np
import pymc3 as pm

config = dict(
    safegraph_data_path = '~/safegraph_data/',
    two_day_test = False,
    grouper = ['county_fips','date_range_start']
)

home = os.path.expanduser('~')

# if config['single_day_test']:
#     files = glob.glob(home+'/safegraph_data/social-distancing/v1/**/02/*/*.csv.gz',recursive=True)
# else:
files = glob.glob(home+'/safegraph_data/social-distancing/v1/**/*.csv.gz',recursive=True)

if config['two_day_test']:
    files = files[0:2]
    
social_dist_df_list = []
for file in files:
    social_dist_df_list.append(pd.read_csv(file, dtype={'origin_census_block_group':str}))

social_dist_df = pd.concat(social_dist_df_list)
social_dist_df = social_dist_df.set_index('origin_census_block_group')

#check county FIPS is just first 5 digits of CBG FIPS - done elsewhere

##aggregate to county level
#get CBG populations
# 2018 ACS data
#TODO: pull all the files for this from FTP, reduce to total pop and concatenate.
# https://www2.census.gov/programs-surveys/acs/replicate_estimates/2018/data/5-year/150/

# cbg_pops = pd.read_csv('data/B01001_01.csv')
# cbg_pops = cbg_pops.dropna(subset=['GEOID'])
# cbg_pops['cbg_fips'] = cbg_pops['GEOID'].apply(lambda x: x[7:])
# cbg_pops= cbg_pops.set_index('cbg_fips')
#social_dist_df.index.difference(cbg_pops.index)

#for now, use safegraph 2016 simplified datasets
cbg_descriptions = pd.read_csv(config['safegraph_data_path']+'safegraph_open_census_data/metadata/cbg_field_descriptions.csv')
sg_cbg_pops = pd.read_csv(config['safegraph_data_path']+'safegraph_open_census_data/data/cbg_b01.csv',dtype={'census_block_group':str})
sg_cbg_pops['county_fips'] = sg_cbg_pops['census_block_group'].apply(lambda x: x[0:5])
sg_cbg_pops = sg_cbg_pops.set_index('census_block_group')
sg_cbg_pops = sg_cbg_pops.rename(columns={'B01001e1':'cbg_population'})
sg_cbg_pops = sg_cbg_pops[['cbg_population','county_fips']] #total population estimate

# checks that indexes mainly match 
social_dist_df.index.difference(sg_cbg_pops.index)

# join census populations to mobility data
social_dist_df = social_dist_df.join(sg_cbg_pops, how='inner')

# Reweight to adjust for sampling bias
# calculate adjustment factor for counts based on CBG population vs panel pop
# TODO: make single call to transform
grouper = config['grouper']
social_dist_df['county_population'] = social_dist_df.groupby(grouper)['cbg_population'].transform(np.sum)
social_dist_df['county_device_count'] = social_dist_df.groupby(grouper)['device_count'].transform(np.sum)
social_dist_df['n_cbg'] = social_dist_df.groupby(grouper)['cbg_population'].transform('count')
social_dist_df['adjustment_factor'] = social_dist_df['cbg_population']/social_dist_df['county_population'] * social_dist_df['county_device_count']/social_dist_df['device_count']


#adjust completely_home_device_count
social_dist_df['adjusted_completely_home_device_count'] = social_dist_df['adjustment_factor'] * social_dist_df['completely_home_device_count']
social_dist_df['pct_staying_home'] = social_dist_df['adjusted_completely_home_device_count']/social_dist_df['device_count']

#calculate weight (cbg population over county population) for weighted averages of summary statistics
social_dist_df['cbg_pop_weight'] = social_dist_df['cbg_population']/social_dist_df['county_population']
social_dist_df['_weighted_median_home_dwell_time_term'] = social_dist_df['cbg_pop_weight'] * social_dist_df['median_home_dwell_time']

#Aggregate to county level
social_dist_county_df = social_dist_df.groupby(grouper).aggregate(
        weighted_median_home_dwell_time =('_weighted_median_home_dwell_time_term', np.sum),
        adjusted_completely_home_device_count=('adjusted_completely_home_device_count', np.sum),
        population = ('cbg_population', np.sum),
        device_count = ('device_count', np.sum)
        )

social_dist_county_df['pct_staying_home'] = social_dist_county_df['adjusted_completely_home_device_count']/social_dist_county_df['device_count']

#TODO: Calculate % at home 

#TODO: build hiearchical model
# Three levels - National, state, county. But what about day? Safegraph uses 7 day rolling, but that screws up weather
# Conclusion: make each day independent. Either seperate models or just duplicate parameters
# Priors? Maximally uninformative or lump a couple weeks in Jan as a baseline?

social_dist_county_df.to_pickle('data/social_dist_county_df.pkl')

n_counties = social_dist_df

with pm.Model() as staying_home_model:
    #Uniformative prior for national
    nat_mu = pm.Beta('phi', alpha=.01, beta=.01)
    nat_nu = pm.Gamma('kappa', mu=1, sigma=1)
    state_mus = pm.Beta(alpha=nat_mu*nat_nu,  beta= (1-nat_mu)*nat_nu, shape=50)
    state_nus = pm.Gamma('kappa', mu=1, sigma=1, shape=50)
    county_thetas = pm.Beta(alpha=state_mus*state_nus, beta= (1-state_mus)*state_nus, shape=)
    y = pm.Binomial('y', n=panel_pop, p=cbg_thetas, observed=completely_home_device_count)

