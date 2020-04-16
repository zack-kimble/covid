import os, glob
import pandas as pd

config = dict(
    safegraph_data_path = '~/safegraph_data/'
)

home = os.path.expanduser('~')
files = glob.glob(home+'/safegraph_data/social-distancing/v1/**/*.csv.gz',recursive=True)

social_dist_df_list = []
for file in files:
    social_dist_df_list.append(pd.read_csv(file))

social_dist_df = pd.concat(social_dist_df_list)

#check county FIPS is just first 5 digits of CBG FIPS

##aggregate to county level
#rebalance completely_home_device_count based on CBG population vs panel pop
#sum completely_home_device_count
#weighted average median_home_dwell_time
#Calculate % at home

#build hiearchical model
# Three levels - National, state, county. But what about day? Safegraph uses 7 day rolling, but that screws up weather
# Conclusion: make each day independent. Either seperate models or just duplicate parameters
# Priors? Maximally uninformative or lump a couple weeks in Jan as a baseline?

