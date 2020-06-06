'''
Deep Q-Learning Network
用gym裡面的小遊戲當作練習
小遊戲：用一部滑輪去平衡木桿
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gym

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9  # 最優選擇動作百分比，有機會不選到最優的
GAMMA = 0.9
TARGET_REPLACE_ITER = 100  # 每100次更新真(實際)的NETWORK參數，可以當作真正領悟的參數
MEMORY_CAPACITY = 2000 # 記憶體大小，當作人類腦記憶容量
env = gym.make('CartPole-v0')   # 立杆子遊戲的環境
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 杆子能做的動作
N_STATES = env.observation_space.shape[0]   # 杆子能獲得的環境信息數

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()     # 正在學習的狀態
        self.target_net = Net()   # 學習到一段落領悟的記憶

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # now state, action, reward, next state
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = LR)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON: # 選擇最大優勢
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action
    
    def store_transition(self, s, a, r, s_):  # state, action, reward, next state
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter += 1

        # 抽取記憶中的記憶
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.FloatTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # 針對做過的動作b_a, 來選q_eval(q_eval原本有所有動作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a) # 丟入現在的狀態可以取得相對應該做的action
        q_next = self.target_net(b_s_).detach() # q_next不做反向傳遞，所以用detach不更新參數
        q_target = b_r + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollection Experience...')
for i in range(400):
    s = env.reset()
    ep_r = 0
    
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify reward 莫凡說原來的reward不太好訓練，因此自己修改了一些，讓車子在越旁邊跟竿子越斜reward越低
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
        
        if done:
            break

        s = s_