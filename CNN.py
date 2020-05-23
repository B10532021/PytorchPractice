import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root = './mnist/', # 保存路徑
    train = True, # MNIST裡面的training data(有分training跟testing)
    transform = torchvision.transforms.ToTensor(), # nparray/pixel -> tensor，圖片值從0~255 -> 0~1
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
)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255. 
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1, 28, 28)
            nn.Conv2d(
                in_channels = 1, # 因為是黑白照片
                out_channels = 16,
                kernel_size = 5, # Filter的size -> 5 * 5
                stride = 1, # 每次Filter移動的距離
                padding = 2  # 在照片旁邊補上的pixel數，padding=(kernel_size-1)/2 if stride=1
            ),  # ->(16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # ->(16, 14, 14)
        )
        self.conv2 = nn.Sequential( # (16, 14, 14)
            nn.Conv2d(
                in_channels = 16, # 因為是黑白照片
                out_channels = 32,
                kernel_size = 5, # Filter的size -> 5 * 5
                stride = 1, # 每次Filter移動的距離
                padding = 2  # 在照片旁邊補上的pixel數，padding=(kernel_size-1)/2 if stride=1
            ), # ->(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), # ->(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10) # CNN後半段會flatten變成fully connected network，輸出結果有10個 0~9
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 這步為flatten將資料變成一維，x.size(0)為batch size，-1的意思是(32, 7, 7) -> 32 * 7 *7
        output = self.out(x)
        return output, x

cnn = CNN()
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# plt.ion()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader): #  normalize x when iterate train_loader
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            # if HAS_SK:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
