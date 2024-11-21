#%%
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

import matplotlib
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device {device}')
# %%
class CosModel(nn.Module) :
    def __init__(self) :
        super(CosModel,self).__init__()
        self.layer1 = nn.Linear(1,16)
        self.layer2 = nn.Linear(16,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,1)
        self.active = nn.ReLU()
        
    def forward(self,x) :
        x = self.active(self.layer1(x))
        x = self.active(self.layer2(x))
        x = self.active(self.layer3(x))
        x = self.layer4(x)
        return x

model = CosModel()
# %%
dataset = torch.load('cos_dataset.pth')
# %%
batch_size = 1000
dataloader = DataLoader(dataset,batch_size)
# %%
epochs = 5000
optimizer = optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()
# %%
for epoch in range(epochs) :
    for X,Y in dataloader :
        y_pred = model(X)
        loss = loss_fn(Y,y_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 500 == 0 :
        print(f"epoch {epoch+1}, loss {loss.item()}")
# %%
x_test  = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
y_test = model(x_test)
plt.plot(x_test,y_test.detach().numpy(),color='blue')
plt.legend()
plt.show()
# %%
