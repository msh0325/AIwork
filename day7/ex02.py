#%% xor 문제 해결하기 
import torch
import torch.nn as nn

# %%
torch.set_printoptions(precision=4)
# %% xor 학습 데이터
X = torch.tensor([[0,0],[1,0],[0,1],[1,1]],dtype=torch.float32) #문제
Y = torch.tensor([[0],[1],[1],[0]],dtype=torch.float32) #답

print(X,Y)
# %% model 만들기
class NeuralNet(nn.Module) : #nn.Module에서 상속받아 선언하겠다 라는 뜻ㅎ
    def __init__(self,input_size,hidden_size,num_classes) :
        super(NeuralNet,self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() #활성함수
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x) : 
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(2,10,1) #2 * 10 = 20개의 파라미터를 가진 신경망
# %% model 이용해 학습 시키기 위한 준비
learning_rate = 0.01 # 학습할 때 얼마만틈 보정할 것인지? 
n_iters = 1000 # 학습을 1000번 시키겠다는 뜻ㅎ

loss = nn.MSELoss() #loss함수. 답지와 얼마나 오차가 있는지
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# %% 학습 시작
for epoch in range(n_iters) : 
    y_predicted = model(X) # X값을 이용해 y를 예측하기
    
    #loss 함수
    l = loss(Y,y_predicted)
    l.backward()
    
    optimizer.step()
    
    optimizer.zero_grad() # 미분한 값 초기화하기. 경사도 0으로 리셋
    
    if(epoch %100==0) : #선택사항. 중간중간 확인용?
        print(l.item())
# %% 검증 시작
pred = model(torch.tensor([1,0],dtype=torch.float32))
print(pred)
# %%
