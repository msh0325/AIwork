#%%
import numpy as np
#%%
x1 = np.random.randint(0,100,10)

print(x1)
# %%
#print([value for value in x1])

x2 = np.array([value for value in x1])
print(x2)
# %%
x3 = np.array([value for value in x1 if value%2]) # 홀수만 배열에 추가하기
print(x3)
# %%
