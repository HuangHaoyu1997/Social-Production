'''
利用RNN-based RL Policy生成符号解析式
控制Social Production任务
'''

import torch
import torch.nn as nn
from cgp import *
from utils import ParentSibling, ComputingTree
from torch.distributions import Categorical
from CartPoleContinuous import CartPoleContinuousEnv
from env import Env

import argparse, math, os, sys, gym
import numpy as np
from gym import wrappers
from function import *
from configuration import config
from CartPoleContinuous import CartPoleContinuousEnv
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils as utils

import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cpu')

env_name = 'SocialProduction'
env = Env()
info = env.reset() # len(info)=23
print(len(info))

env.seed(config.seed)                                                 # 随机数种子
torch.manual_seed(config.seed)                                        # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)

def s0():
    '''返回env状态的第0维度'''
    return state[0]
def s1():
    '''返回env状态的第1维度'''
    return state[1]
def s2():
    '''返回env状态的第2维度'''
    return state[2]
def s3():
    '''返回env状态的第3维度'''
    return state[3]

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
]

class lstm(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layer):
        super(lstm,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hn=None, cn=None):
        if hn is not None and cn is not None:
            x, (hn, cn) = self.lstm(x, (hn, cn))
        else: x, (hn, cn) = self.lstm(x)
        s, b, h = x.size() # s序列长度, b批大小, h隐层维度
        # print(s,b,h,hn.shape,cn.shape)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        x = torch.softmax(x, dim=-1)
        return x, hn, cn

def policy_evaluator(tau, env, func_set, episode=config.Epoch):
    '''
    policy evaluation
    policy is represented by a symbol sequence `tau`
    episode: test the policy for `config.Epoch` times, and average the episode reward
    '''
    global state
    r_epi = 0
    for i in range(episode):
        s = env.reset()
        state = s
        done = False
        reward = 0
        count = 0
        while not done:
            action = ComputingTree(tau, func_set)
            s, r, done, _ = env.step(np.array([action]))
            state = s
            reward += r
            count += 1
            if count >= config.max_step: break
        r_epi += reward
    return r_epi / episode

def policy_generator(model, func_set,):
    '''
    return a sequence of symbols
    '''
    func_dim = len(func_set) # dimension of function set / categorical distribution
    tau = [] # symbol sequence

    # generte tau_1 with empty parent and sibling
    [iP, iS], P, S = ParentSibling(tau, func_set)
    PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
    
    counter = 1
    log_prob = 0
    joint_entropy = 0
    hn, cn = torch.zeros(2,1,config.hidden_size), torch.zeros(2,1,config.hidden_size)
    while counter > 0:
        phi, hn, cn = model(PS, hn, cn)
        
        mask = ApplyConstraints(tau, func_set)
        phi_after_mask = phi * mask
        phi_after_mask = phi_after_mask / phi_after_mask.sum()
        # print(phi,'\n',phi_after_mask,'\n\n')

        dist = Categorical(phi_after_mask[0,0])
        new_op = dist.sample()
        # print(new_op, phi_after_mask[0,0,new_op].log())
        log_prob += phi_after_mask[0,0,new_op].log()
        joint_entropy += dist.entropy()
        tau.append(new_op.item())
        
        PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
        counter += func_set[new_op].arity - 1
        if counter==0: break
        if len(tau) > config.N_COLS: return -1, 0, 0
        [iP, iS], P, S = ParentSibling(tau, func_set)
    
    if (func_dim-1 not in tau) and (func_dim-2 not in tau) and (func_dim-3 not in tau) and (func_dim-4 not in tau):
        return -1, 0, 0
    
    return tau, log_prob, joint_entropy

def ApplyConstraints(tau, func_set):
    '''
    给RNN输出的categorical概率施加约束
    如果parent是log/exp,则exp/log的概率为0
    如果parent是sin/cos,则cos/sin的概率为0
    '''
    # 如果tau空集合,不能选择常量作为根节点
    if len(tau)==0:
        mask = torch.tensor([0 if func_set[i].name in ['s0','s1','s2','s3'] else 1 for i in range(len(func_set))])
        return mask
    
    # 如果tau非空
    else:
        # iP是将要生成的node的parent在tau中的idx
        [iP,iS], P, S = ParentSibling(tau, func_set)
        # parent是iP在func_set中的idx
        parent = tau[iP]
        if func_set[parent].name == 'sin':
            mask = torch.tensor([0 if func_set[i].name=='cos' else 1 for i in range(len(func_set))])
        elif func_set[parent].name == 'cos':
            mask = torch.tensor([0 if func_set[i].name=='sin' else 1 for i in range(len(func_set))])
        elif func_set[parent].name == 'log':
            mask = torch.tensor([0 if func_set[i].name=='exp' else 1 for i in range(len(func_set))])
        elif func_set[parent].name == 'exp':
            mask = torch.tensor([0 if func_set[i].name=='log' else 1 for i in range(len(func_set))])
        else:
            mask = torch.ones(len(func_set))
        return mask

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

    def update_parameters(self, rewards, log_probs, entropies, gamma):# 更新参数
        
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

dir = './results/ckpt_' + env_name
if not os.path.exists(dir):    
    os.mkdir(dir)

for i_episode in range(config.num_episodes):
    entropies = []
    log_probs = []
    rewards = []
    for t in range(config.batch): # 1次生成10个tau,分别测试
        tau, log_prob, entropy = agent.symbolic_generator()
        # print(tau, log_prob, entropy)
        reward = policy_evaluator(tau, env, func_set)

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
    agent.update_parameters(rewards, log_probs, entropies, config.gamma)

    if i_episode % config.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-'+str(i_episode)+'.pkl'))
    
    print("Episode: {}, reward: {}".format(i_episode, np.mean(rewards)))

env.close()
