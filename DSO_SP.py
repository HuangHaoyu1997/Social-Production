'''
利用RNN-based RL Policy生成符号解析式
控制Social Production任务
'''

import torch
import torch.nn as nn
from cgp import *
from utils import info_parser
from env import Env
from DSO import lstm, policy_evaluator, policy_generator
import math, os, sys
import numpy as np
from function import *
from configuration import config
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils as utils

import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cpu')

env_name = 'SocialProduction'
env = Env()
info = env.reset() # len(info)=23
state = info_parser(info)

env.seed(config.seed)                                                 # 随机数种子
torch.manual_seed(config.seed)                                        # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)

def s0(): return state[0]
def s1(): return state[1]
def s2(): return state[2]
def s3(): return state[3]
def s4(): return state[4]
def s5(): return state[5]
def s6(): return state[6]
def s7(): return state[7]
def s8(): return state[8]
def s9(): return state[9]
def s10():return state[10]
def s11():return state[11]
def s12():return state[12]
def s13():return state[13]
def s14():return state[14]
def s15():return state[15]
def s16():return state[16]
def s17():return state[17]
def s18():return state[18]
def s19():return state[19]
def s20():return state[20]
def s21():return state[21]
def s22():return state[22]

func_set = [
    Function(op.add, 2),        # 0
    Function(op.sub, 2),        # 1
    Function(op.mul, 2),        # 2
    Function(protected_div, 2), # 3
    Function(math.sin, 1),      # 4
    Function(math.cos, 1),      # 5
    Function(ln, 1),      # 6
    Function(exp, 1),      # 7
    Function(const_01, 0),      # 8
    # Function(const_1, 0),       # 9
    # Function(const_5, 0),       # 10
    Function(s0, 0),
    Function(s1, 0),
    Function(s2, 0),
    Function(s3, 0),
    Function(s4, 0),
    Function(s5, 0),
    Function(s6, 0),
    Function(s7, 0),
    Function(s8, 0),
    Function(s9, 0),
    Function(s10, 0),
    Function(s11, 0),
    Function(s12, 0),
    Function(s13, 0),
    Function(s14, 0),
    Function(s15, 0),
    Function(s16, 0),
    Function(s17, 0),
    Function(s18, 0),
    Function(s19, 0),
    Function(s20, 0),
    Function(s21, 0),
    Function(s22, 0),
]


class REINFORCE:
    def __init__(self, func_set, hidden_size):
        '''
        func_set: 符号库
        
        '''
        self.model = lstm(input_size = 2*len(func_set),
                                hidden_size = hidden_size, 
                                output_size = len(func_set), 
                                num_layer = 2
        )
        # self.model = self.model.cuda()                              # GPU版本
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2) # 优化器
        self.fs = func_set
        self.model.train()

    def symbolic_generator(self):
        tau = -1
        while tau == -1:
            tau, log_prob, entropy = policy_generator(self.model, self.fs)
        print('done')
        return tau, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies):# 更新参数
        loss = 0
        # print(rewards)
        for i in reversed(range(len(rewards))):
            R = Variable(torch.tensor(rewards[i]))
            loss = loss - log_probs[i]*R - 0.01*entropies[i]
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 10)             # 梯度裁剪，梯度的最大L2范数=40
        self.optimizer.step()

agent = REINFORCE(func_set, config.hidden_size)

dir = './results/DSO_' + env_name
if not os.path.exists(dir):    
    os.mkdir(dir)

for i_episode in range(config.num_episodes):
    entropies = []
    log_probs = []
    rewards = []
    for t in range(config.batch): # 1次生成10个tau,分别测试
        tau, log_prob, entropy = agent.symbolic_generator()
        # print(tau, log_prob, entropy)
        reward = policy_evaluator(tau, env, func_set, episode=config.Epoch, env_name=env_name)

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)

    # 截取reward排前90%的样本
    length = int(config.batch*(1-config.epsilon))
    idx = np.array(rewards).argsort()[::-1][:length]
    
    rewards = np.array(rewards)[idx]
    entropies = np.array(entropies)[idx]
    log_probs = np.array(log_probs)[idx]

    # 1局游戏结束后开始更新参数
    agent.update_parameters(rewards, log_probs, entropies)

    if i_episode % config.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'dso_sp-'+str(i_episode)+'.pkl'))
    
    print("Episode: {}, reward: {}".format(i_episode, np.mean(rewards)))

env.close()
