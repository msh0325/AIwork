#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
# %%
data_1 = pd.read_csv('..\datasheet\lotto_1.csv') #1~600
data_2 = pd.read_csv('..\datasheet\lotto_2.csv') #601~

data = pd.concat([data_1,data_2],ignore_index=True)
# %%
data.info() #iso는 인덱스, c1~cb까지가 당첨 번호. win은 상금인듯
# %% 당첨금액의 ~~원 써있는거 지우기. 데이터 안쓰지만 해보기
price_colums = ["win1_pric","win2_pric","win3_pric","win4_pric","win5_pric"]
for col in price_colums :
    print(f"conver colum {col}")
    data[col] = data[col].str.replace('원','').str.replace(',','').astype(np.int64) 
    # replace를 이용해 원을 만나면 지우고, 컴마를 만나도 지운다
   
data.info() 
# %%
data.head()
# %% 각 자리당 숫자 분포도 그리기
plt.figure(figsize=(10,6))
data[['c1','c2','c3','c4','c5','c6','cb']].hist(bins=50)
plt.show()
# %% 분포도를 이용하여 랜덤하게 추측하기
colums = ['c1','c2','c3','c4','c5','c6']
recomandations = {} 

for col in colums :
    most_common_number = data[col].value_counts().idxmax()
    #가장 많이 나온 번호 뽑기
    recomandations[col] = most_common_number

print(recomandations)
    
# %% 직접 가중치를 찾아서 넣어주는 중

def weighted_random_choice(column) :
    value_count = data[column].value_counts()
    numbers = value_count.index.tolist()
    weights = value_count.values.tolist()
    
    chosen_number = random.choices(numbers,weights=weights,k=1)
    #가중치가 높은 것을 우선으로 랜덤하게 뽑음, 하나만 뽑음
    return chosen_number[0]

# %%
for col in colums :
    recomandations[col] = weighted_random_choice(col)

print("추천번호 : ",recomandations)
# %%
