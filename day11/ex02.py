#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# %%
data_1 = pd.read_csv('..\datasheet\lotto_1.csv') #1~600
data_2 = pd.read_csv('..\datasheet\lotto_2.csv') #601~

data = pd.concat([data_1,data_2],ignore_index=True)
# %%
data.info()

# %% iso를 index로 바꾸기
data.set_index('iso',inplace=True)
data.sort_index(inplace=True)

data.head()
# %%
lotto_numbers = data[['c1','c2','c3','c4','c5','c6','cb']]
lotto_numbers.head()
# %% 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(lotto_numbers)
# %%
print(scaled_data)
print(scaled_data.shape)
# %%
def create_sequence(data,seqeunce_length) :
    sequences = []
    targets = []
    for i in range(len(data)-seqeunce_length) :
        seq = data[i: i+seqeunce_length] #문제
        target = data[i+seqeunce_length] #정답
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences),np.array(targets)
# %%
sequence_length = 10

x,y = create_sequence(scaled_data,sequence_length)
# %%
print(x.shape)
print(y.shape)
# %% sklearn을 이용해 학습용과 테스트용 데이터셋 만들기.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,shuffle=False)
#test size는 전체의 2할만 테스트로 쓰겠다는 뜻
x_train = torch.tensor(x_train,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32)
#그냥은 못쓰니까 tensor로 바꾸기
# %%
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)
# %%
