'''
讓神經網路學習畫曲線
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5   # GENERATE的靈感數量
ART_COMPONENT = 15  # 組成OUTPUT的要素數量
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENT) for _ in range(BATCH_SIZE)])

# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():
    '''
    create BATCH_SIZE of curve lines to be the real art works
    '''
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENT)
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENT, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid() # 為了要判斷是否像真的的概率
)

opt_G = torch.optim.Adam(G.parameters(), lr = LR_G)
opt_D = torch.optim.Adam(D.parameters(), lr = LR_D)

plt.ion() # 用來畫連續的圖片、影像

for epoch in range(1):
    artist_paintings = artist_works()
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))
    G_paintings = G(G_ideas)

    # 判斷是否是真的圖的機率
    prob_true_painting = D(artist_paintings)
    prob_draw_painting = D(G_paintings)

    #  D_loss 要增加真實化的可能性，要降低畫出來的圖的可能性
    #  因此如果prob_draw_painting越高的話，要讓Discriminator知道是假的 => 1 - prob可以越少
    D_loss = - torch.mean(torch.log(prob_true_painting) + torch.log(1 - prob_draw_painting))
    #  G_loss 要讓畫出來的越像真的，所以要增加prob_draw_painting
    G_loss = torch.mean(torch.log(1 - prob_draw_painting))

    opt_D.zero_grad()
    print(prob_draw_painting)
    D_loss.backward(retain_graph = True)
    opt_D.step()


    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

#     if step % 50 == 0:  # plotting
#         plt.cla()
#         plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
#         plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
#         plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
#         plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
#         plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
#         plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

# plt.ioff()
# plt.show()