#%%
for i in range(1,10) : #1에서 10 전까지 찍기
    print(i)
# %%
for i in range(10) : #0부터 10 전까지 찍기
    print(i)
# %%
for i in range(1,10,2) : #1부터 2씩 늘려가며 10 전까지 찍기
    print(i)
# %%
for i in range(10,0,-1) : #10에서 1까지 역순으로 찍기
    print(i)
# %%
_list = [3,6,9]
for i in _list : #list 하나씩 순회하며 찍기
    print(i)
for i in range(3) : 
    print(_list[i])
# %%
for i,d in enumerate(_list) : #list 출력 시 인덱스와 같이 출력 
    print(i,d) 
# %%
