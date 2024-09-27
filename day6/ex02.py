#%% loadtxt 함수
import numpy as np

#%%
gate_data = np.loadtxt('./gate.csv',delimiter=',',dtype=np.int8,skiprows=1)
print(gate_data)
# %%
