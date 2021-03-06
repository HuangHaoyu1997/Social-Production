'''
Vanilla Policy Gradient for Continuous Control Env
'''
import argparse, math, os, sys, gym, torch, pickle, time
import numpy as np
from gym import wrappers
from function import *
from configuration import config
from CartPoleContinuous import CartPoleContinuousEnv
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from configuration import config
from utils import *
from env import Env
from collections import Counter
matplotlib.use('pdf') # 不显示图片,直接保存pdf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 运行时间e.g.'2022-04-07-15-10-12'

run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19] # 运行时间e.g.'2022-04-07-15-10-12'


env_name = 'SocialProduction'
env = Env()

if config.display:
    env = wrappers.Monitor(env, './result/policygradient/{}-experiment'.format(env_name), force=True)

env.seed(config.seed)                                                 # 随机数种子
torch.manual_seed(config.seed)                                        # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)

class Policy(nn.Module):                                            # 神经网络定义的策略
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space                            # 动作空间
        num_outputs = 2                         # 动作空间的维度

        self.linear1 = nn.Linear(num_inputs, hidden_size)           # 隐层神经元数量
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
        self.linear3_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        a = F.softplus(self.linear3(x), beta=0.0005)                                  # 为了输出连续域动作，policy net定义了
        b = F.softplus(self.linear3_(x), beta=0.0005)                                 # 一个多维Beta分布，维度=动作空间的维度
        # torch.nn.Softplus()
        # a += Variable(torch.tensor(1e-2))
        # b += Variable(torch.tensor(1e-2))
        # if a.item()<=1e-2: 
        # if b.item()<=1e-2: 
        return a, b

class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)    # 创建策略网络
        # self.model = self.model.cuda()                              # GPU版本
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr) # 优化器
        self.model.train()


    def select_action(self, state):
        # mu, sigma_sq = self.model(Variable(state).cuda())
        a, b = self.model(Variable(state))
        
        beta = torch.distributions.Beta(a,b)
        sample = beta.sample()
        action = [sample[0,0].item()*50, sample[0,1].item()*4+1] # 最低工资[0,50],最高工资倍率[1,5]
        
        log_prob = beta.log_prob(sample)
        entropy = beta.entropy()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):# 更新参数
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]                                # 倒序计算累计期望
            # loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
            # print(len(rewards), len(log_probs))
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (0.1*entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)             # 梯度裁剪，梯度的最大L2范数=40
        self.optimizer.step()

agent = REINFORCE(config.hidden_size, env.observation_shape, env.action_shape)

dir = './results/ckpt_' + env_name
if not os.path.exists(dir):    
    os.mkdir(dir)



for i_episode in range(config.num_episodes):
    tick = time.time()
    done = False
    # seed = i_episode; env.seed(seed)
    info = env.reset()
    state = torch.Tensor([info_parser(info)])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(config.num_steps): # 1个episode最长持续的timestep
        action, log_prob, entropy = agent.select_action(state)
        info, reward, done = env.step( action ) # np.zeros((config.N1))
        state = torch.Tensor([info_parser(info)])
        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        '''for event_point in config.event_point:
            if abs(env.t - event_point) <= config.event_duration: # t%100 == 99:
                env.event_simulator('GreatDepression')'''
        if done: env.render(str(np.round(np.sum(rewards),2))); break # 只保存文件，不画图
        
    # 1局游戏结束后开始更新参数
    agent.update_parameters(rewards, log_probs, entropies, config.gamma)

    if i_episode % config.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'VPG_SP-'+str(i_episode)+'.pkl'))

    print("Episode: {}, reward: {}, time: {}".format(i_episode, np.sum(rewards), time.time()-tick))

