import torch
from torch.autograd import Variable # torch 再輸入跟輸出的時候要將型態變成Variable
import torch.nn.functional as F # 裡面包含了torch的一些函數，ex：activation function
import matplotlib.pyplot as plt

# 製造假資料
# 先創100個(x, y) = (1, 1)的基本資料
# 最後concat兩組資料->一組資料
nData = torch.ones(100, 2)
x0 = torch.normal(2 * nData, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * nData, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # torch的float型式
y = torch.cat((y0, y1)).type(torch.LongTensor) # torch的64 bits integer

# 畫圖
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


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
        self.out = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        """
        定義這個network的架構是如何傳遞的
        正向傳播輸入值，network分析出輸出值
        """
        x = F.relu(self.hidden(x))
        output = self.out(x)

        return output

# 創建net
# 可以用print(net)印出架構
# input有兩個features，output也有兩個種類
"""
Net (
  (hidden): Linear (2 -> 10)
  (predict): Linear (10 -> 2)
)
"""
net = Net(n_feature=2, n_hidden=10, n_output=2)

# 定義optimizer(梯度下降的優化器)和learning rate，這邊使用stochastic gradient descent
# 定義使用的loss function，這邊使用cross entropy loss
# 算誤差的時候, 注意真實值不是 one-hot 形式的, 而是1D Tensor, (batch,)
# 但是預測值是2D tensor (batch, n_classes)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
loss_func = torch.nn.CrossEntropyLoss()


Epoch = 100
for time in range(Epoch):
    output = net(x) # network預測出的值
    loss = loss_func(output, y) # 計算誤差

    optimizer.zero_grad() # 清空上一步殘留的更新參數值
    loss.backward() # back propagation，計算參數更新值
    optimizer.step() # 將參數更新值施加到net的節點參數上

    if time % 2 == 0:
        plt.cla()
        # 經過了一道 softmax 的激勵函數後的最大概率才是預測值
        prediction = torch.max(F.softmax(output), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.  # 預測中有多少和真實值一樣
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.5)

plt.ioff()  # 停止畫圖
plt.show()

# 第二種搭建network的方法
# 直接使用pytorch的內建model
# 與上面的network異曲同工之妙
# net2 = torch.nn.Sequential(
#     torch.nn.Linear(2, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 2)
# )
