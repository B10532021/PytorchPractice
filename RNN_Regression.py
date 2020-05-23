'''
用sin(x)當作Input預測cos(x)的值
'''
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyper Parameters
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# show data
steps = np.linspace(0, np.pi * 2, 100, dtype = np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE, 
            hidden_size = 32,
            num_layers = 1, 
            batch_first = True
        )
        self.out = nn.Linear(32, 1)
    
    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [] # 紀錄每個timestep的output
        for time_step in range(r_out.size(1)):   # 計算每次的output
            outs.append(self.out(r_out[:, time_step, :]))

        return torch.stack(outs, dim = 1), h_state
        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        
        # or even simpler, since nn.Linear can accept inputs of any dimension 
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs

rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
loss_func = nn.MSELoss()


h_state = None # 第一次的h_state不會有東西
for step in range(60):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))  # newaxis的目的是把(timestep) -> (batch, timestep, inputsize)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

