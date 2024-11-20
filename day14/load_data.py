#%%
import torch
from torch.utils.data import Dataset,DataLoader

from my_dataset import XorDataset
print("module load complete")
print(f"torch version : {torch.__version__}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device : {device}")
# %%
dataset = torch.load('xordataset.pth')
print(dataset)
# %%
dataloader = DataLoader(dataset,1,shuffle=False)
for idx,(X,Y) in enumerate(dataloader) :
    print(X,Y)
# %%
