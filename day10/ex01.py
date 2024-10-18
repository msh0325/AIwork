#%%
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
import math

print(torch.__version__)
# %%
x = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
y = torch.sin(x)
# %%
class SimpleSinModel(nn.Module) :
    def __init__(self) :
        super(SimpleSinModel,self).__init__()
        
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,1)
        self.relu = nn.ReLU()
    
    def forward(self,x) :
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
simple_model = SimpleSinModel()

# %%
learning_rate = 0.01
epochs = 5000
optimizer = optim.Adam(simple_model.parameters(),lr=learning_rate)
loss_fn = nn.MSELoss()
# %%
loss_value=[]
for epoch in range(epochs) :
    optimizer.zero_grad()
    y_pred = simple_model(x)
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()
    
    loss_value.append(loss.item()) #그냥 넣으면 텐서값으로 나와서 텐서를 일반으로 바꿔주는 item함수
    if epoch %500 ==0 :
        print(loss.item())
# %%
plt.figure(figsize=(10,5))
plt.plot(range(epochs),loss_value,label='loss',color ='green')
plt.legend()
plt.show()
# %%
x_test = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
y_test = simple_model(x_test)

plt.figure(figsize=(10,5))
plt.plot(x_test,y_test.detach().numpy(),color='blue')
plt.legend()
plt.show()

# %%
