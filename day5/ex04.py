#%% 3d array sort
import numpy as np
#%%
x1 = np.array([
     [[101,131]],
      [[103,141]],
      [[107,181]],
      [[102,111]],
      [[100,191]]
])

print(x1)
# %%
print(x1[:,0,0]) # 첫번째 행
print(x1[:,0,1]) # 두번째 행
# %% 
sort_indice1 = np.argsort(x1[:,0,0]) # argsort로 첫번째 행 인덱스 뽑기
sort_indice2 = np.argsort(x1[:,0,1]) # argsort로 두번째 행 인덱스 뽑기

print(x1[:,0,0][sort_indice1]) # 정렬된 인덱스로 첫번째 행 정렬
print(x1[:,0,1][sort_indice2]) # 정렬된 인덱스로 두번째 행 정렬
# %%
