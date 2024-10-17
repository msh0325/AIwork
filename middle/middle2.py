#%%
import numpy as np
import pandas as pd
import os
# %%
raw_df = pd.read_csv('datasheet\owid-covid-data.csv')
raw_df.info()
# %%
raw_df['location'].unique()
# %%
revise_df = raw_df[['date','total_cases','location','population']]
france_df = revise_df[revise_df['location']=='France']
korea_df = revise_df[revise_df['location']=='South Korea']
france_df.info()
korea_df.info()
# %%
france_df.head()
# %%
korea_df.head()
# %%
france_data_index_df = france_df.set_index('date')
france_data_index_df.head()
# %%
korea_data_index_df = korea_df.set_index('date')
korea_data_index_df.head()
# %%
final_df = pd.DataFrame(
    {
        'Korea' : korea_data_index_df['total_cases']*_rate,
        'France' : france_data_index_df['total_cases'] 
    }, index = korea_data_index_df.index
)
# %%
final_df.head()
# %%
final_df.plot.line(rot=45)
# %%
france_population = france_data_index_df['population']['2022-01-01']
korea_population = korea_data_index_df['population']['2022-01-01']
# %%
_rate = round(france_population/korea_population,2)
# %%
