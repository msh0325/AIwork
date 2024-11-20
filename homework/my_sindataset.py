#%%
import torch
import math
from torch.utils.data import DataLoader,Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')
#%%
class SinDataset(Dataset) :
    def __init__(self,x,y) :
        self.x_data = x
        self.y_data = y
        
    def __len__(self) :
        return len(self.x_data)
    
    def __getitem__(self,idx) :
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x,y   
# %%
