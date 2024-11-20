#%%
import torch
from torch.utils.data import Dataset,DataLoader

print("module load complete")
print(f"torch version : {torch.__version__}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device : {device}")
# %%
class XorDataset(Dataset) :
    def __init__(self,x,y) : #생성자
        self.x_data = x
        self.y_data = y
        
    def __len__(self) : #dataset 길이
        return len(self.x_data)
    
    def __getitem__(self,idx) : #item 가져오기
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x,y
    #위의 세개가 필수 항목