import pandas as pd
import glob, os

config = dict(
    safegraph_data_path = '~/safegraph_data/'
)



joined_df = pd.read_pickle('data/us_data_with_latent_populations.pkl')

#social distancing data

sg = pd.read_csv(config['safegraph_data_path']+'social-distancing/v1/2020/01/22/2020-01-22-social-distancing.csv.gz',
                 dtype={'origin_census_block_group':str})

core = pd.read_csv('/home/zack/safegraph_data/core/2020/03/CoreRecords-CORE_POI-2019_03-2020-03-25.zip')

home_panel = pd.read_csv('/home/zack/safegraph_data/weekly-patterns/v1/home_summary_file/2020-04-05-home-panel-summary.csv')

cbg_fips_codes = pd.read_csv('/home/zack/safegraph_data/safegraph_open_census_data/metadata/cbg_fips_codes.csv')

#trend data

joined_df = joined_df.reset_index()
joined_df['county_fips'] = joined_df['UID'].apply(lambda x: str(int(x))[3:])
joined_df = joined_df.set_index(['county_fips', 'date'])

sg['county_fips'] = sg['origin_census_block_group'].apply(lambda x: x[0:5])
sg['date'] = pd.to_datetime(sg['date_range_start'], utc=True).dt.date
sg = sg.set_index(['county_fips','date'])
test= sg.join(joined_df, how='inner')


joined_df.loc[17133]
joined_df.loc[10150]
joined_df.loc["01001"]

sg['origin_census_block_group'].sort_values()

type(sg.date_range_start[0])

type(test[0])

joined_df['FIPS'].astype(int)

x = pd.Series([99999, 100, 60])
df = pd.DataFrame.from_dict({'column1':['a','b','c'],'floats':[1.3,4.5,8.3]})
df.dtypes
df['test'] = x

joined_df.index.get_level_values('date')
sg.index.get_level_values('date')

joined_df.index.get_level_values('county_fips')
sg.index.get_level_values('county_fips')

sg.index.difference(joined_df.index)

missing = joined_df[joined_df["Admin2"]=="Bibb"]
fips = joined_df["FIPS"].sort_values()
joined_df.loc[1007]


social_dist_df['county_fips'] = social_dist_df['origin_census_block_group'].apply(lambda x: int(str(x)[0:5]))
social_dist_df['date'] = pd.to_datetime(social_dist_df['date_range_start'], utc=True).dt.date
social_dist_df = social_dist_df.set_index(['county_fips','date'])
epidemiology_social_dist_df = social_dist_df.join(joined_df)