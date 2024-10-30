#%%
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
# %%
w = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

x = torch.linspace(0,10,1000).unsqueeze(1)
y = 3*x+4
# %%
learning_rate = 0.01
epochs = 5000
optimizer = optim.SGD([w,b],lr = learning_rate)
loss_fn = nn.MSELoss()
# %%
for epoch in range(epochs):
    hx = x*w+b
    loss = loss_fn(hx,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch %500 == 0 :
        print(loss.item())
    
# %%
x_test = torch.linspace(0,10,1000).unsqueeze(1)
y_pred = w*x_test+b

plt.figure(figsize=(10,5))
plt.plot(x_test,y_pred.detach().numpy(),color='blue')
plt.legend()
plt.show()
# %%
