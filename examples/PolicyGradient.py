'''
Vanilla Policy Gradient for Continuous Control Env
'''
import argparse, math, os, sys, gym, torch
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

env_name = 'CartPoleContinuous'
env = CartPoleContinuousEnv()

if config.display:
    env = wrappers.Monitor(env, './result/policygradient/{}-experiment'.format(env_name), force=True)

env.seed(config.seed)                                                 # 随机数种子
torch.manual_seed(config.seed)                                        # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)

class Policy(nn.Module):                                            # 神经网络定义的策略
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space                            # 动作空间
        num_outputs = action_space.shape[0]                         # 动作空间的维度

        self.linear1 = nn.Linear(num_inputs, hidden_size)           # 隐层神经元数量
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        a = F.softplus(self.linear2(x))                                  # 为了输出连续域动作，实际上policy net定义了
        b = F.softplus(self.linear2_(x))                                 # 一个多维Beta分布，维度=动作空间的维度

        return a, b

class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)    # 创建策略网络
        # self.model = self.model.cuda()                              # GPU版本
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2) # 优化器
        self.model.train()


    def select_action(self, state):
        # mu, sigma_sq = self.model(Variable(state).cuda())
        a, b = self.model(Variable(state))
        beta = torch.distributions.Beta(a,b)
        sample = beta.sample()
        action = (sample*2 - 1).item() # 定义域[-1,1]
        log_prob = beta.log_prob(sample)
        entropy = beta.entropy()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):# 更新参数
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]                                # 倒序计算累计期望
            # loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (0.001*entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 10)             # 梯度裁剪，梯度的最大L2范数=40
        self.optimizer.step()

agent = REINFORCE(config.hidden_size, env.observation_space.shape[0], env.action_space)

dir = './results/ckpt_' + env_name
if not os.path.exists(dir):    
    os.mkdir(dir)

for i_episode in range(config.num_episodes):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(config.num_steps): # 1个episode最长持续的timestep
        action, log_prob, entropy = agent.select_action(state)
        next_state, reward, done, _ = env.step(np.array([action]))

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break
    # 1局游戏结束后开始更新参数
    agent.update_parameters(rewards, log_probs, entropies, config.gamma)


    if i_episode % config.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-'+str(i_episode)+'.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

env.close()

