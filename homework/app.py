#%%
import torch
import math
from torch.utils.data import DataLoader,Dataset

from my_sindataset import SinDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')
# %%
import matplotlib
import matplotlib.pyplot as plt 
#%% sin data 생성
X = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
Y = torch.sin(X)
#%% cos data 생성
X = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
Y = torch.cos(X)
# %%
print(X.shape)
print(Y.shape)
# %%
dataset = SinDataset(X,Y)
# %%
batch_size = 1
dataloader = DataLoader(dataset,batch_size,shuffle=False)
# %% sin dataset 저장
torch.save(dataset,'sin_dataset.pth')
#%% cos dataset 저장
torch.save(dataset,'cos_dataset.pth')
# %%
plt.plot(X,Y,color='blue')
plt.legend()
plt.show()
# %%
