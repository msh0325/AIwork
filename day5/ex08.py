#%% shape
import numpy as np
# %%
x1 = np.random.randint(0,100,16)
print(x1)
print(x1.shape)
# %%
_x1 = np.expand_dims(x1,axis=0) # 0번째 축에 차원 하나 추가
print(_x1)
print(_x1.shape)
#%%
_x1 = np.expand_dims(x1,axis = 1) # 첫번째 축에 차원 하나 추가
print(_x1)
print(_x1.shape)
# %%
x2 = np.reshape(x1,(4,4))
print(x2)
print(x2.shape)
# %%
x3 = np.reshape(x1,(-1,2)) # -1의 뜻 : 일단 2개씩 묶고 되면 자동으로 채워짐
print(x3)
# %%
x4 = np.reshape(x1,(4,-1)) # 그럼 얘는 일단 4묶음을 만들고 딱 떨어지면 채워짐
print(x4)
# %%
