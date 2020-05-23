# 一樣使用MNIST當做訓練資料
# RNN可以看作從一張照片由上到下看的經過
# 照片由上到下代表時間的順序
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEPS = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root = './mnist/', # 保存路徑
    train = True, # MNIST裡面的training data(有分training跟testing)
    transform = transforms.ToTensor(), # nparray/pixel -> tensor，圖片值從0~255 -> 0~1
    download = DOWNLOAD_MNIST # 是否要下載
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE, 
    shuffle = True, 
)
# 畫出一張範例
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


test_data = torchvision.datasets.MNIST(
    root = './mnist/', # 保存路徑
    train = False, # MNIST裡面的testing data(有分training跟testing)
    transform=transforms.ToTensor()
)
test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:2000]/255. 
test_y = test_data.test_labels.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # 這邊使用LSTM而非RNN是因為莫煩說RNN的結果不太好
        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,   # input的順序可以為(batch, time_step, input)或是(time_step, batch, input)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        '''
        r_out是rnn目前的理解結果
        h_n, h_c是rnn支線跟主線的記憶跟理解，會在下一次跟新的input一起做分析
        x 是(batch, timestep, input_size)
        '''
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :]) # (batch, timestep, input)要取最後一個timestep的結果
        return out

rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader): #  normalize x when iterate train_loader
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# primt 10 prediction from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')