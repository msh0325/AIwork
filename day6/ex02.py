#%% loadtxt 함수
import numpy as np

#%%
gate_data = np.loadtxt("D:\\aiwork_msh\\AIwork\\day6\\gate.csv",delimiter=',',dtype=np.int8,skiprows=1)
print(gate_data)
# %%
