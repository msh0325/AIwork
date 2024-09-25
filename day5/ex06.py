#%%
import numpy as np

# %%
x1 = np.random.randint(0,100,10)
x2 = np.random.randint(0,100,10)

print(x1,x2)
# %%
x3 = np.array([_v for _v in zip(x1,x2)]) # zip을 이용해 1차원 배열 두개를 2차원 배열로 만듬
print(x3)
# %%
x4 = np.array([_v for _v in zip(x1,x2) if _v[0]<50]) # 첫번째 행의 값이 50 미만인 애들로만  만듬
print(x4)
# %%
