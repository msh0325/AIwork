#%% sin 함수 예측하기
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math # sin 함수 받기 위한것
import matplotlib
import matplotlib.pyplot as plt
# %%
x = torch.linspace(0,2*math.pi,1000).unsqueeze(1) # 0부터 2파이까지 1000조각으로 나눔
y = torch.sin(x)

# %%
print(x)
# %%
print(y)
# %%
plt.figure(figsize=(10,5))
plt.plot(x,y,label='sin(x)',color='blue')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()
# %% sin 함수 예측하기
model = nn.Sequential(
    
    nn.Linear(1,8),
    nn.ReLU(),
    nn.Linear(8,1)
    
)

# %%
learing_rate = 0.01
epochs = 100
optimizer = optim.Adam(model.parameters(),lr=learing_rate)
loss_fn = nn.MSELoss()
# %%
for epoch in range(epochs) :
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 :
        print(f"epoch {epoch} : loss {loss}")

# %% 검증
x_test = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
y_test = model(x_test)

plt.figure(figsize=(10,5))
plt.plot(x_test,y_test,label='model(x)',color='blue')
plt.xlabel('x')
plt.ylabel('model(x)')
plt.legend()
plt.show()
# %%
