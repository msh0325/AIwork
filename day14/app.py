#%%
import torch
from torch.utils.data import Dataset,DataLoader

from my_dataset import XorDataset
print("module load complete")
print(f"torch version : {torch.__version__}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device : {device}")
# %%
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]],dtype=torch.float32)
# %%
dataset = XorDataset(X,Y)
#%% batch_size를 이용해 한번에 가져오는 데이터 수를 조절할 수 있음. 
dataloader = DataLoader(dataset,batch_size=1,shuffle=False)

for idx,(X,Y) in enumerate(dataloader) : 
    print(f"batch idx : {idx}")
    print(f"x : {X}")
    print(f"y : {Y}")
# %%
dataloader = DataLoader(dataset,batch_size=4,shuffle=False)
for idx,(X,Y) in enumerate(dataloader) : 
    print(f"batch idx : {idx}")
    print(f"x : {X}")
    print(f"y : {Y}")
# %% save dataset
torch.save(dataset,"xordataset.pth") #xordataset.pth 파일로 저장
# %%
