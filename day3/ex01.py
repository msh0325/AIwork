#%%
_a = lambda x : x**2 #x에 값 넣으면 제곱해서 반환

print(_a(5))
# %%
_sum = lambda x,y : x+y

print(_sum(5,6))
# %% 람다 자체 반환? 크루즈?크루저?
def _inc(n) : 
    return lambda x : x+n

_inc3 = _inc(3)

print(_inc3(6))
# %%
_inc2 = _inc(2)

print(_inc2(6))

# %% 매핑과 람다함수를 이용해 for문없이 각 값 더하기
a = [1,2,3]
b = [4,5,6]

print(list(map(lambda x,y : x+y,a,b)))
# %% 
_list = [5,8,2,6,1,9,3,7,4]

_newlist = list(map(lambda x : 0 if x<3 else 1,_list)) # x<3일때 0, 아니라면 1 넣기
print(_newlist)
# %%
_list = [5,8,2,6,1,9,3,7,4]

_newlist = list(filter(lambda x : x>3,_list)) # x>3일때만 _newlist에 넣기. 작은건 걸러서 안넣기
print(_newlist)
# %%
