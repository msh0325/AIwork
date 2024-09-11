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
# %% _list의 3,4만 뽑기 1
_,_,a,b,_,_ = _list
print(a,b)
# %% _list의 3,4만 뽑기 2
_,_,a,b,*_ = _list
print(a,b)
# %% _list의 3,4만 뽑기 3
_,_,*a,_,_ = _list
print(a)
# %%
_list = [1,2,3]
# %%
_list.append(77) #list 뒤에 추가하기
#%%
print(_list)
# %%
_list.insert(0,88) #원하는 위치에 추가하기
# %%
_list.insert(-1,88) #맨 끝에서 앞으로 한칸??
# %%
_list.insert(1,88) #두번째 넣기
# %%
print(_list.pop()) #스택처럼 마지막에 넣은거 빼기
# %%
print(_list.pop(0)) #큐처럼 처음에 넣은거 빼기
# %%
