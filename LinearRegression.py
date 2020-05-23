import torch
from torch.autograd import Variable # torch 再輸入跟輸出的時候要將型態變成Variable
import torch.nn.functional as F # 裡面包含了torch的一些函數，ex：activation function

# 製造假資料
# unsqueeze的用意在將一維資料包成二維的，因為input data只能接受2維，[1, 2, 3, 4] => [[1, 2, 3, 4]]
# x data(tensor), shape(100, 1)
# y data(tensor), shape(100, 1)
# 將x, y變成Variable型態
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1) 
y = x.pow(2) + 0.2 * torch.rand(x.size()) 

x, y = Variable(x), Variable(y)


# 開始搭建Net，需要創一個Net的class，並繼承torch.nn.Module
# 同時必須創建__init__()跟forward()
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        """
        包含一些基礎訊息，像是hidden layer、predict layer
        並定義layer的形式
        n_feature, n_hidden, n_output為自訂參數
        """
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        """
        定義這個network的架構是如何傳遞的
        正向傳播輸入值，network分析出輸出值
        x為input
        """
        x = F.relu(self.hidden(x))
        output = self.predict(x)

        return output

# 創建net
# 可以用print(net)印出架構
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""
net = Net(n_feature=1, n_hidden=10, n_output=1)

# 定義optimizer(梯度下降的優化器)和learning rate，這邊使用stochastic gradient descent
# 定義使用的loss function，這邊使用mean square error
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
loss_func = torch.nn.MSELoss()


Epoch = 100
for time in range(Epoch):
    prediction = net(x) # network預測出的值
    loss = loss_func(prediction, y) # 計算誤差

    optimizer.zero_grad() # 清空上一步殘留的更新參數值
    loss.backward() # back propagation，計算參數更新值
    optimizer.step() # 將參數更新值施加到net的節點參數上

    if time % 5 == 0:
        print('epoch:', time, '  loss:', loss)

# save(有兩種方法)
torch.save(net, 'net.pkl') # 保存整個net(架構跟參數...)
torch.save(net.state_dict(), 'net_param.pkl') # 只保存net的節點參數

# load(有兩種方法)
net1 = torch.load('net.pkl')
# 只load參數, 要先取得原本的模型架構
net2 = Net(n_feature=1, n_hidden=10, n_output=1)
net2.load_state_dict(torch.load('net_param.pkl'))