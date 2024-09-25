#%%
import numpy as np

# %%
x1 = np.random.randint(0,100,10)

print(x1)
# %%
print("sum : ",x1.sum()) # x1의 합
print("avg : ",x1.mean()) # x1의 평균
print("standard : ",x1.std()) # x1의 표준편차
print("var : ",x1.var()) # x1의 분산
# %%
x2 = np.array([
    np.random.randint(0,100,10),
    np.random.randint(0,100,10)
])

print(x2)
# %%
print(x2.sum())
# %%
