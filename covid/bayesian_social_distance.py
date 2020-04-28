import pandas as pd
import numpy as np
import pymc3 as pm
import theano
import pickle

#TODO: build hiearchical model
# Three levels - National, state, county. But what about day? Safegraph uses 7 day rolling, but that screws up weather
# Conclusion: make each day independent. Either seperate models or just duplicate parameters
# Priors? Maximally uninformative or lump a couple weeks in Jan as a baseline?

social_dist_df = pd.read_pickle('data/social_dist_and_county_raw.pkl')
social_dist_df = social_dist_df.loc[:,['date','state_fips','county_fips','device_count','completely_home_device_count']]
social_dist_df = social_dist_df.loc[social_dist_df['date'] == pd.to_datetime('2020-03-01')]

print('getting unique state values')
unique_states = social_dist_df['state_fips'].unique()
n_states = len(unique_states)
state_lookup =  dict(zip(unique_states, range(n_states)))
state_idx = social_dist_df['state_fips'].map(state_lookup).values

print('getting unique county values')
unique_counties = social_dist_df['county_fips'].unique()
n_counties = len(unique_counties)
county_lookup = dict(zip(unique_counties, range(n_counties)))
county_idx = social_dist_df['county_fips'].map(county_lookup).values


unique_states_counties = social_dist_df[['state_fips','county_fips']].drop_duplicates()
state_to_county_idx = unique_states_counties['state_fips'].map(state_lookup).values
device_count = social_dist_df['device_count']
completely_home_device_count = social_dist_df['completely_home_device_count']



with pm.Model() as staying_home_model:
    #Uniformative prior for national
    nat_mu = pm.Beta('nat_mu', alpha=.01, beta=.01)
    nat_nu = pm.Gamma('nat_kappa', mu=1, sigma=1)
    state_mus = pm.Beta('state_mu', alpha=nat_mu*nat_nu,  beta= (1-nat_mu)*nat_nu, shape=n_states)
    state_nus = pm.Gamma('state_kappa', mu=1, sigma=1, shape=n_states)
    county_thetas = pm.Beta('theta',alpha=state_mus[state_to_county_idx] * state_nus[state_to_county_idx],
                            beta= (1-state_mus[state_to_county_idx]) * state_nus[state_to_county_idx],
                            shape=n_counties)
    y = pm.Binomial('y', n=device_count, p=county_thetas[county_idx], observed=completely_home_device_count)

print('starting MCMC')
with staying_home_model:
    trace = pm.sample(750, tune=200)

with open('data/county_trace.pkl','wb') as f:
    pickle.dump(trace, f)
    
with open('data/county_trace.pkl','rb') as f:
    trace = pickle.load(f)

pm.traceplot(trace)
import arviz as az
az.plot_trace(trace, var_names=['nat_mu','nat_kappa'])
az.plot_forest(trace, var_names=['state_mu'], r_hat=True, combined=True)
az.plot_forest(trace, var_names=['state_kappa'], r_hat=True, combined=True)
az.plot_forest(trace, var_names=['theta'], r_hat=True, combined=False)

import seaborn as sns

thetas = trace.get_values('theta', combine=False)
thetas = np.array(thetas)
thetas[:,:,0]
thetas[...,0]

for chain in thetas[...,2]:    
    sns.distplot(chain)


state_mus = np.random.random(n_states)
county_thetas = state_mus[state_to_county_idx] + range(0,n_counties)
binom_p = county_thetas[county_idx]

social_dist_df["binom_p"] = binom_p

theano.config.profile = True 
theano.config.profile_memory = True 
staying_home_model.profile(staying_home_model.logpt).summary()