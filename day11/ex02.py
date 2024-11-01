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
class LottoLSTM(nn.Module) :
    def __init__ (self,input_size,hidden_size,output_size,num_layers):
        super(LottoLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
    
    def forward(self,x) : 
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out
    
model = LottoLSTM(7,64,7,2)
# %%
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
# %% train
num_epochs = 1000
for epoch in range(num_epochs) :
    model.train()
    outputs = model(x_train)
    loss = loss_fn(outputs,y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch %100 == 0 :
        print(f"epoch [{epoch}/{num_epochs}], loss {loss.item():.4f}")
# %% test
model.eval()
with torch.no_grad() :
    test_outputs = model(x_test)
    test_loss = loss_fn(test_outputs,y_test)
    print(f'Test loss {test_loss.item():.4f}')
# %% 어떤 번호를 찍어주는지 출력
x_tensor = torch.tensor(x,dtype=torch.float32)
last_sequence = x_tensor[-1].unsqueeze(0)
print(last_sequence)
# %%
with torch.no_grad() :
    next_pred = model(last_sequence)

# %%
pre_nums = scaler.inverse_transform(next_pred.detach().numpy())
print(np.round(pre_nums[0]).astype(int)+1)
# %%
