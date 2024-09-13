#pip intall numpy
#터미널에 입력해 넘파이 설치.
#%%
import sys
import numpy as np
print(sys.version)
print(np.__version__)
#%%
a = np.empty(0)
print(a)
# %%
a =  np.append(a,1)
print(a)

# %%
_list1 = [1,2,3,4]
arry1 = np.array(_list1)

print(arry1)
# %%
print(arry1.shape) # 출력값은 튜플
# %%
print(arry1[0:2]) # 0번째(처음)부터 2개만 출력
# %%
print(type(arry1)) # arry1의 타입 출력
print(arry1.dtype) # arry1의 데이터 타입 출력
# %%
_arry1 = arry1.astype(np.float32) # 타입 바꿀 때 float 뒤에 숫자도 붙여야 댐
print(_arry1.dtype)
# %%
arry2 = np.array([1,2,3])
arry3 = np.array([4,5,6])

print(arry2 + arry3) # 텐서연산 지원. list는 뒤에 붙이는게 끝. 얘는 각각 더해줌
# %%
