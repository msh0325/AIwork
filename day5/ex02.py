#%% sort
import numpy as np
# %%
x1 = np.random.randint(0,100,10) #0~99사이에서 10개 랜덤
print(x1)
# %%
print(np.sort(x1)) # 원본에 변화를 줌
print(x1)
# %%
print(np.argsort(x1)) # 원본에 변화 x. 인덱스 번호로 알려줌
print(x1)
sort_indice = np.argsort(x1)
#%%
print(x1[0])
print(x1[sort_indice[0]]) #argsort 이용한 min값 찍기
print(x1[x1.argmin()]) # argmin 이용한 min값 찍기
print(x1[x1.argmax()]) # argmax 이용한 max값 찍기
# %%
x2 = np.array([0,0,0,0,0,0,0,0,0,0])
print(x1[x2])
# %%
print(x1[sort_indice])
# %%
