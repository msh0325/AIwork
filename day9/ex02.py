#%%
import numpy as np
import pandas as pd
import os

# %%
raw_df = pd.read_csv('..\datasheet\survey_results_public.csv')
raw_df.info()
# %% column 이름 보기
for col in raw_df.columns :
    print(col)
# %%
reversed_df = raw_df[['Age','Country','LearnCode',
                      'LanguageHaveWorkedWith',
                      'LanguageWantToWorkWith']]
reversed_df.head()
# %%
print(reversed_df['Age'])
# %% 겹치는거 무시하고 총 몇가지의 연령대가 있는지 체크
print(reversed_df['Age'].drop_duplicates())
# %%
size_by_age = reversed_df.groupby("Age").size()
print(size_by_age)
# %%
size_by_age.plot.bar()
# %% 그래프 순서 reindex 하기
reindex_size_by = size_by_age.reindex(index=[
    "Under 18 years old",
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65 years or older",
    "Prefer not to say"
])
#%% 기본적인 세로막대 그래프
reindex_size_by.plot.bar()
# %% 가로막대 그래프
reindex_size_by.plot.barh()
# %% 파이 모양 그래프
reindex_size_by.plot.pie()
# %% 어느 나라에 더 많은 개발자가 있는가
size_by_country = reversed_df.groupby("Country").size()
size_by_country.head()
# %%
size_by_country.plot.pie()
# %% 나라가 너무 많으니 상위 20개국만 보게 하기
size_by_country.nlargest(20).plot.pie()
# %% 어떤 언어가 젤 많이 쓰였는가
languages = reversed_df["LanguageHaveWorkedWith"].str.split(';',expand=True)
print(languages)
# %%
languages.head()
# %% 항목이 여러개로 나뉘어 있어서 groupby 못쓴대용. 그래서 stack에 넣고 하나씩 보기
size_by_language = languages.stack().value_counts()
size_by_language.head()
# %%
size_by_language.plot.pie()
# %% 상위 10개 언어만 보기
size_by_language.nlargest(10).plot.pie()
# %%
