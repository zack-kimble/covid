import pandas as pd
import numpy as np
import pymc3 as pm
import theano
import pickle
import seaborn as sns

#TODO: build hiearchical model
# Three levels - National, state, county. But what about day? Safegraph uses 7 day rolling, but that screws up weather
# Conclusion: make each day independent. Either seperate models or just duplicate parameters
# Priors? Maximally uninformative or lump a couple weeks in Jan as a baseline?

social_dist_df = pd.read_pickle('data/social_dist_and_county_raw.pkl')
social_dist_df = social_dist_df.loc[:,['date','state_fips','county_fips','device_count','completely_home_device_count']]
social_dist_df = social_dist_df.loc[social_dist_df['date'] < pd.to_datetime('2020-01-05')]

# #create various combined id's to make lookup easier
# social_dist_df['county_fips_date'] = social_dist_df['county_fips'] + social_dist_df['date'].astype(str)
# social_dist_df['state_fips_date'] = social_dist_df['state_fips'] + social_dist_df['date'].astype(str)

# print('getting unique date values')
# unique_dates = social_dist_df['date'].unique()
# n_dates = len(unique_dates)
# date_lookup =  dict(zip(unique_dates, range(n_dates)))

# print('getting unique state values')
# unique_states = social_dist_df['state_fips'].unique()
# n_states = len(unique_states)
# state_lookup =  dict(zip(unique_states, range(n_states)))

# print('getting unique county values')
# unique_counties = social_dist_df['county_fips'].unique()
# n_counties = len(unique_counties)
# county_lookup = dict(zip(unique_counties, range(n_counties)))
# county_idx = social_dist_df['county_fips'].map(county_lookup).values

# print('getting unique date/state values')
# unique_dates_states = social_dist_df[['state_fips_date', 'state_fips','date']].drop_duplicates()
# n_dates_states = len(unique_dates_states)
# date_to_state_idx = unique_dates_states['date'].map(date_lookup).values
# #for next section
# dates_states_lookup = dict(zip(unique_dates_states['state_fips_date'], range(n_dates_states)))

# print('getting unique date/county values') #State is implied because we use county fips which contains state info
# #making combined identifier to make replacement easier
# unique_dates_states_counties = social_dist_df[['county_fips_date','county_fips','state_fips_date']].drop_duplicates()
# n_dates_states_counties = len(unique_dates_states_counties)
# date_to_state_to_county_idx = unique_dates_states_counties['state_fips_date'].map(dates_states_lookup).values
# #for next section
# dates_states_counties_lookup =  dict(zip(unique_dates_states_counties['county_fips_date'], range(n_dates_states_counties)))

# print('make last index for cbg to county lookup')
# date_to_state_to_county_to_cbg_index = social_dist_df['county_fips_date'].map(dates_states_counties_lookup).values


# device_count = social_dist_df['device_count']
# completely_home_device_count = social_dist_df['completely_home_device_count']



# with pm.Model() as staying_home_model:
#     #Uniformative prior for national
#     date_mus = pm.Beta('date_mus', alpha=100, beta=100, shape=n_dates)
#     date_nus = pm.Gamma('date_kappa', mu=80, sigma=5, shape=n_dates)
#     state_mus = pm.Beta('state_mu', alpha=date_mus[date_to_state_idx] * date_nus[date_to_state_idx],  beta= (1-date_mus[date_to_state_idx])*date_nus[date_to_state_idx], shape=n_dates_states)
#     state_nus = pm.Gamma('state_kappa', mu=50, sigma=5, shape=n_dates_states) #variance within states is independent between states
#     county_thetas = pm.Beta('theta',alpha=state_mus[date_to_state_to_county_idx] * state_nus[date_to_state_to_county_idx],
#                             beta= (1-state_mus[date_to_state_to_county_idx]) * state_nus[date_to_state_to_county_idx],
#                             shape=n_dates_states_counties)
#     y = pm.Binomial('y', n=device_count, p=county_thetas[date_to_state_to_county_to_cbg_index], observed=completely_home_device_count)

#Non-centered hiearchical model with logistic link function (essentially GLM with just an intercept)
# with pm.Model() as staying_home_model:
#     #National prior
#     date_mus = pm.Normal('date_mus', mu=-.8, sigma= 1, shape=n_dates)
#     date_sigmas = pm.HalfNormal('date_sigmas', sigma=.17, shape = n_dates) #half normal because we don't actually want very fat tails, since everything gets squished by logistic function anyway
#     state_offsets = pm.Normal('state_offsets', mu=0, sd=1, shape=n_dates_states)
#     state_mus = pm.Deterministic('state_mus', date_mus[date_to_state_idx] + state_offsets * date_sigmas[date_to_state_idx])
#     state_sigmas = pm.HalfNormal('state_sigmas', sigma=.15, shape = n_dates_states) #variance within states is independent between states
#     county_offsets = pm.Normal('county_offsets', mu=0, sd=1, shape=n_dates_states_counties)
#     county_betas = pm.Deterministic('county_betas', state_mus[date_to_state_to_county_idx] + county_offsets * state_sigmas[date_to_state_to_county_idx])
#     county_thetas = pm.Deterministic('county_thetas', pm.math.sigmoid(county_betas))
#     y = pm.Binomial('y', n=device_count, p=county_thetas[date_to_state_to_county_to_cbg_index], observed=completely_home_device_count)

# Single day model
#create various combined id's to make lookup easier
social_dist_df['county_fips_date'] = social_dist_df['county_fips'] + social_dist_df['date'].astype(str)
social_dist_df['state_fips_date'] = social_dist_df['state_fips'] + social_dist_df['date'].astype(str)

print('restrict to single day')
social_dist_df =  social_dist_df.loc[social_dist_df['date'] == pd.to_datetime('2020-01-01')]

print('getting unique state values')
unique_states = social_dist_df['state_fips'].unique()
n_states = len(unique_states)
states_lookup =  dict(zip(unique_states, range(n_states)))

print('getting unique county values')
unique_counties = social_dist_df['county_fips'].unique()
n_counties = len(unique_counties)
county_lookup = dict(zip(unique_counties, range(n_counties)))
county_idx = social_dist_df['county_fips'].map(county_lookup).values

# print('getting unique date/state values')
# unique_dates_states = social_dist_df[['state_fips_date', 'state_fips','date']].drop_duplicates()
# n_dates_states = len(unique_dates_states)
# date_to_state_idx = unique_dates_states['date'].map(date_lookup).values
# #for next section
# dates_states_lookup = dict(zip(unique_dates_states['state_fips_date'], range(n_dates_states)))

print('getting unique state/county values') #State is implied because we use county fips which contains state info
#making combined identifier to make replacement easier
unique_states_counties = social_dist_df[['county_fips','state_fips']].drop_duplicates()
n_states_counties = len(unique_states_counties)
state_to_county_idx = unique_states_counties['state_fips'].map(states_lookup).values
#for next section
states_counties_lookup =  dict(zip(unique_states_counties['county_fips'], range(n_states_counties)))

print('make last index for cbg to county lookup')
state_to_county_to_cbg_index = social_dist_df['county_fips'].map(states_counties_lookup).values


device_count = social_dist_df['device_count']
completely_home_device_count = social_dist_df['completely_home_device_count']

#Daily non-centered hiearchical model with logistic link function (essentially GLM with just an intercept)
with pm.Model() as staying_home_model:
    state_mus = pm.Normal('date_mus', mu=-.8, sigma= 1, shape=n_states)
    state_sigmas = pm.HalfNormal('state_sigmas', sigma=.15, shape = n_states) #variance within states is independent between states
    county_offsets = pm.Normal('county_offsets', mu=0, sd=1, shape=n_states_counties)
    county_betas = pm.Deterministic('county_betas', state_mus[state_to_county_idx] + county_offsets * state_sigmas[state_to_county_idx])
    county_thetas = pm.Deterministic('county_thetas', pm.math.sigmoid(county_betas))
    y = pm.Binomial('y', n=device_count, p=county_thetas[state_to_county_to_cbg_index], observed=completely_home_device_count)


print('starting MCMC')
with staying_home_model:
    trace = pm.sample()

with open('data/county_trace.pkl','wb') as f:
    pickle.dump(trace, f)


with staying_home_model:
    prior_sample = pm.sample_prior_predictive()


    
def logistic(array):
    return 1/(1+np.exp(-1*array))
        

for date_mu in prior_sample['date_mus']:
    sns.kdeplot(logistic(date_mu))
 
for date_sigma in prior_sample['date_sigmas']:
    sns.distplot(date_sigma, kde=False)    
    
for state_offset in prior_sample['state_offsets']:
    sns.kdeplot(state_offset)    
    
for state_mu in prior_sample['state_mus']:
    sns.distplot(state_mu,kde=False)   

for state_mu in prior_sample['state_mus']:
    sns.distplot(logistic(state_mu),kde=False)   

for state_mu in prior_sample['state_mus']:
    sns.kdeplot(logistic(state_mu))   

mus_iter = iter([x for x in prior_sample['state_mus']])

sns.kdeplot(logistic(next(mus_iter)))

sns.kdeplot(logistic(prior_sample['state_mus'].flatten()))

logistic(prior_sample['date_mus'][0])

prior_sample['date_sigmas'][3]
np.mean(prior_sample['date_sigmas'])




with open('data/county_trace.pkl','wb') as f:
    pickle.dump(trace, f)
    
with open('data/county_trace.pkl','rb') as f:
    trace = pickle.load(f)

pm.traceplot(trace)
import arviz as az

data = az.from_pymc3(prior=prior_sample)
az.plot_density(data,group='prior')

az.plot_trace(trace, var_names=['date_mus'])
az.plot_forest(trace, var_names=['date_mus'])
az.plot_forest(trace, var_names=['state_mu'], r_hat=True, combined=True)
az.plot_forest(trace, var_names=['state_kappa'], r_hat=True, combined=True)
az.plot_forest(trace, var_names=['theta'], r_hat=True, combined=False)

az.plot_posterior(trace, var_names='county_thetas', combined=False)

import seaborn as sns
sns.kdeplot(prior_sample['y'])

date_mus = trace.get_values('date_mus',combine=False)
date_mus = np.array(date_mus)
date_mus.shape

date_sigmas = trace.get_values('date_sigmas',combine=False)
date_sigmas = np.array(date_sigmas)
date_sigmas.shape

state_offsets = trace.get_values('state_offsets',combine=False)
state_offsets = np.array(state_offsets)
state_offsets.shape

state_sigmas = trace.get_values('state_sigmas',combine=False)
state_sigmas = np.array(state_sigmas)
state_sigmas.shape

county_offsets = trace.get_values('county_offsets',combine=False)
county_offsets = np.array(county_offsets)
county_offsets.shape



for chain in state_offsets:
    print(chain.shape)
    print( np.mean(np.mean(chain, axis=1)))
    print(np.mean(np.std(chain, axis=1)))


for chain in county_offsets:
    print(chain.shape)
    print( np.mean(np.mean(chain, axis=1)))
    print(np.mean(np.std(chain, axis=1)))

for chain in state_sigmas:
    print(state_sigmas.shape)
    print( np.mean(np.mean(state_sigmas, axis=1)))
    print(np.mean(np.std(state_sigmas, axis=1)))



np.mean(state_offsets[0],axis=1)
np.std(state_offsets[0],axis=0)

trace.stat_names
sum(trace.get_sampler_stats('diverging'))

rhat = pm.stats.rhat(trace)

for key in rhat.keys():
    mean = np.mean(rhat[key])
    print(f'{key}: {mean}')


    
for chain in     

thetas = trace.get_values('county_thetas', combine=False)

thetas = np.array(thetas)
thetas.shape
thetas[:,:,0].shape
thetas[...,0].shape

for chain in thetas[...,5]:    
    sns.distplot(chain)


# state_mus = np.random.random(n_states)
# county_thetas = state_mus[state_to_county_idx] + range(0,n_counties)
# binom_p = county_thetas[county_idx]

# social_dist_df["binom_p"] = binom_p

#Test indexing
date_mus = unique_dates.astype(str)
state_mus = np.char.add(date_mus[date_to_state_idx], unique_dates_states['state_fips'].values)
county_thetas = np.char.add(state_mus[date_to_state_to_county_idx], unique_dates_states_counties['county_fips'].values)
cbg_p = np.char.add(county_thetas[date_to_state_to_county_to_cbg_index], social_dist_df.index.values)
social_dist_df['pymc3_indexed_value'] = cbg_p
social_dist_df['index_check_value'] = social_dist_df['date'].astype(str) + social_dist_df['state_fips'] + social_dist_df['county_fips'] + social_dist_df.index.values
social_dist_df['values_match'] = social_dist_df['pymc3_indexed_value'] == social_dist_df['index_check_value']
np.all(social_dist_df['values_match'])


theano.config.profile = True 
theano.config.profile_memory = True 
staying_home_model.profile(staying_home_model.logpt).summary()

x = np.linspace(0,1)
y = np.exp(pm.Beta.dist(alpha=3, beta=5, shape=n_dates).logp(x).eval())
sns.lineplot(x,y)

import scipy.stats

scipy.stats.beta(2,1).pdf(x)

x1 = np.linspace(0,10)
y1 = np.exp(pm.Gamma.dist( mu=1, sigma=1, shape=n_dates).logp(x).eval())

sns.lineplot(x1,y1)

state_kappas = pm.Gamma.dist('state_kappa', mu=30, sigma=10, shape=1800)

dir(state_kappas)
sum(state_kappas.random(100) == 0)

date_mus = pm.Beta.dist( alpha=2, beta=2, shape=n_dates).random(1)
date_nus = pm.Gamma.dist( mu=20, sigma=5, shape=n_dates).random(1)
state_mus = pm.Beta.dist( alpha=date_mus[date_to_state_idx] * date_nus[date_to_state_idx],  beta= (1-date_mus[date_to_state_idx])*date_nus[date_to_state_idx], shape=n_dates_states)

sum(state_mus.random(100)-1==0)

with staying_home_model:
    x = state_mus.random()
    y = state_nus.random()
result = [sum(x==1),sum(x==0),sum(y==0)]
print(result)