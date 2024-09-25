#%% 2d array sort
import numpy as np
# %%
x1 = np.array([
    [1,9],
    [2,8],
    [0,7],
    [3,6],
    [5,4]]
)

print(x1)
# %%
_index = [0,1,2,3,4]
print(x1[:,0])
print(x1[_index,0])
# %%
print(x1[:,0].argsort()) # 첫번째라인[:,0] sort
_sort_indice1 = x1[:,0].argsort()
print(x1[:,0][_sort_indice1]) # sort 출력하기
_sort_indice2 = x1[:,1].argsort()
print(x1[:,1][_sort_indice2]) # sort 출력하기
# %%
