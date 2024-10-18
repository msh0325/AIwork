#%%
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
# %%
x = torch.linspace(0,100,1000)
y = 2*x+3
# %%
class simpleLineModel(nn.Module) :
    def __init__(self) : 
        super(simpleLineModel)