#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
x = np.linspace(0,10,50) # 선형함수 내의 0~10사이의 랜덤한 수 50개
y = 2*x+(1+np.random.randn(50))

print(x,y)
# %%
plt.scatter(x,y,label = 'data point') # scatter 그래프 그리기
plt.show

# %% 데이터를 가지고 함수 그리기
A = np.vstack([x,np.ones(len(x))]).T
print(A)
a,c = np.linalg.lstsq(A,y,rcond=None)[0]
print(f"기울기 {a} , 절편 {c}")
# %%
plt.scatter(x,y,label='data point')
plt.plot(x,a*x+c,'r',label='fitting line') # 예측값이 얼마나 잘나오는지 표시
plt.legend()
plt.show()
# %%
## 파이토치로 간단한 신경망 만들기???? linear 이용하기? 찾아보기