#%%
import numpy as np
import pandas as pd

pd.show_versions()
# %% 파일 경로 통해 데이터값 불러오기
raw_df = pd.read_csv("D:\\aiwork_msh\\AIwork\\datasheet\\owid-covid-data.csv")
# %% 값 잘 불러와졌는지 확인
raw_df.head() #67개 값의 column
# %% column 이름 보기
raw_df.info()
# %% 필요한 값만 가져와서 추출하기
revise_df = raw_df[
    ["iso_code",
     "location",
     "date",
     "total_cases",
     "population"
    ]]
revise_df.head()
# %% 데이터 내의 총 국가 수
location = revise_df["location"].unique() #location에서 중복된 국가 이름 지움
print(location)
# %% south korea 추출하기
korea_df = revise_df[revise_df["location"] == "South Korea"]
korea_df.head()
# %% date 기준으로 보기위해 인덱스 넣기
korea_date_index_df = korea_df.set_index('date')
korea_date_index_df.head()
# %% 그래프 그리기 위한 데이터 추출
kor_total_case = korea_date_index_df["total_cases"] # index가 이미 date기 때문에 이거만 해도 댐
kor_total_case.head()
# %% 그래프 그리기
kor_total_case.plot()
# %% 위랑 똑같이 USA도 추출하기
usa_df = revise_df[revise_df["location"] == "United States"]
usa_df.head()    
# %%
usa_date_idnex_df = usa_df.set_index('date')
usa_date_idnex_df.head()
# %%
usa_total_case = usa_date_idnex_df["total_cases"]
usa_total_case.head()
# %% usa 그래프 끝
usa_total_case.plot()
# %% KOR 그래프와 USA 그래프 비교하기 위해 합치기
final_df = pd.DataFrame( #딕셔너리 이용해 합침
    {
        'kor' : kor_total_case,
        'usa' : usa_total_case
    },index = kor_total_case.index
)
final_df.head()
# %% 합친 데이터 이용해 그래프 그리기
final_df.plot(rot=45)
# 미국의 그래프가 높아 훨씬 위험해 보이지만, 총인구수가 다르기 때문에
# 정규화 작업 필요
# %% 정규화? 작업시작. 인구수 가져오기
usa_population = usa_date_idnex_df['population']['2022-01-01']
print(usa_population)

kor_population = korea_date_index_df['population']['2022-01-01']
print(kor_population)
# %% 비율 계산하기. round 사용해서 소수점 둘째자리까지 표시
_rate = round(usa_population/kor_population,2)
print(_rate)
# %% 찐최종 데이터 합친 후 그래프 그리기
_final_df = pd.DataFrame(
    {
        'kor' : _rate*kor_total_case, #위에서 구한 비율 이용하기
        'usa' : usa_total_case
    },index = korea_date_index_df.index
)

_final_df.plot(rot=45)
# %%
