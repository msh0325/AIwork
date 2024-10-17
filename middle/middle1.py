#%%
import numpy as np
import pandas as pd
import os
# %%
raw_df = pd.read_csv('datasheet\survey_results_public.csv')
# %%
raw_df.head()
# %%
for col in raw_df.columns :
    print(col)
# %%
reversed_df = raw_df[['Age']]
print(reversed_df)
# %%
print(reversed_df.drop_duplicates())
# %%
size_by_age = reversed_df.groupby('Age').size()
# %%
size_by_age.plot.bar()
# %%
reindex_by_age = size_by_age.reindex(index=[
    "Under 18 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65 years or older",
    "Prefer not to say"]
)
# %%
reindex_by_age.plot.bar()
# %%
