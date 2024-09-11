#%% unpack
_list = [1,2,3,4,5,6]

a = _list[0]
b = _list[1]
print(a,b)
# %%
*a, = _list
print(a)
a,b,c,d,_,_ = _list
print(a,b)
# %%
a,b,c,d,*_ = _list # d 이후로 생략
print(c,d)
# %%
*a, = _list
