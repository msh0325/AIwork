#%%
_tuple = (1,2,3,4,5,6)
print(_tuple)
#%%
print(_tuple[0])
#_tuple[0] = 1 # tuple은 안됨
# %%
_tuple = (v for v in range(10)) #tuple은 이렇게 초기화 불가능?
print(_tuple)
# %%
_list = [1,2,3,4,5,6]
print(_list)
print(_list[2])
_list[2] = 7 # list는 됨
print(_list[2])
# %%
_list = [v for v in range(10)] # list는 이렇게 초기화 가능
print(_list)
# %%
__list = [_list]
print(__list)
# %%
print(__list[0])
print(__list[0][1])
# %%
_list1 = [1,3,5,7,9]
_list2 = [2,4,6,8,10]

print(_list1 + _list2) # list를 그냥 합친거
# %%
poly = [(x+y) for x,y in zip(_list1,_list2)] # list의 원소끼리 합한것
print(poly)
# %%
