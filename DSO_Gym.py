'''
利用RNN-based RL Policy生成符号解析式
'''

import torch
import torch.nn as nn
from cgp import *
from DSO import lstm, ParentSibling, ComputingTree
from torch.distributions import Categorical
from CartPoleContinuous import CartPoleContinuousEnv

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

# env_name = 'CartPoleContinuous'
# env = CartPoleContinuousEnv()
env_name = 'LunarLander'
env = gym.make('LunarLander-v2')
obs_dim = env.observation_space.shape[0]
state = env.reset()

env.seed(config.seed)                                                 # 随机数种子
torch.manual_seed(config.seed)                                        # Gym、numpy、Pytorch都要设置随机数种子
np.random.seed(config.seed)

def s(index):
    return state[index]


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
def s4():
    '''返回env状态的第4维度'''
    return state[4]
def s5():
    '''返回env状态的第5维度'''
    return state[5]
def s6():
    '''返回env状态的第6维度'''
    return state[6]
def s7():
    '''返回env状态的第7维度'''
    return state[7]

func_set = [
    Function(op.add, 2),        # 0
    Function(op.sub, 2),        # 1
    Function(op.mul, 2),        # 2
    Function(protected_div, 2), # 3
    Function(math.sin, 1),      # 4
    Function(math.cos, 1),      # 5
    Function(ln, 1),      # 6
    Function(exp, 1),      # 7
    # Function(const_01, 0),      # 8
    # Function(const_1, 0),       # 9
    # Function(const_5, 0),       # 10
]
state_f = [Function(s(index=i), 0, name='s'+str(i)) for i in range(obs_dim)]
func_set.extend(state_f)
for f in state_f:
    print(f.name)

def policy_evaluator(tau, env, func_set, episode, env_name):
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
        count = 0
        while not done:
            action = ComputingTree(tau, func_set, env_name)
            s, r, done, _ = env.step(np.array(action))
            state = s
            count += 1
            if count >= config.max_step: break
            r_epi += r
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
        mask = torch.tensor([0 if func_set[i].name in [f.name for f in state_f] else 1 for i in range(len(func_set))])
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
        
        self.model = lstm(input_size = 2*len(func_set),
                                hidden_size = hidden_size, 
                                output_size = len(func_set), 
                                num_layer = 2
        )
        # self.model = self.model.cuda()                              # GPU版本
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3) # 优化器
        self.fs = func_set
        self.model.train()

    def symbolic_generator(self):
        tau = -1
        while tau == -1:
            tau, log_prob, entropy = policy_generator(self.model, self.fs)
        # print('done')
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

def load_test():
    model_param = torch.load('./results/ckpt_CartPoleContinuous/reinforce-100.pkl')
    model = lstm(input_size = 2*len(func_set),
                                    hidden_size = 128, 
                                    output_size = len(func_set), 
                                    num_layer = 2
            )
    model.load_state_dict(model_param)
    agent = REINFORCE(func_set, 128)
    agent.model = model
    for _ in range(10):
        tau, _, _ = agent.symbolic_generator()
        print(tau)

if __name__ == '__main__':
    agent = REINFORCE(func_set, config.hidden_size)

    dir = './results/DSO_' + env_name
    if not os.path.exists(dir):    
        os.mkdir(dir)

    for i_episode in range(config.num_episodes):
        entropies = []
        log_probs = []
        rewards = []
        rr = -1000
        tt = None
        for t in range(config.batch): # 1次生成10个tau,分别测试
            tau, log_prob, entropy = agent.symbolic_generator()
            
            # print(tau, log_prob, entropy)
            reward = policy_evaluator(tau, env, func_set, config.Epoch,env_name)
            if reward>rr: 
                rr = reward
                tt = tau

            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
        print(rr,tt)
        # 截取reward排前60%的样本
        length = int(config.batch*(1-config.epsilon))
        idx = np.array(rewards).argsort()[::-1][:length]
        
        rewards = np.array(rewards)[idx]
        entropies = np.array(entropies)[idx]
        log_probs = np.array(log_probs)[idx]


        # 1局游戏结束后开始更新参数
        agent.update_parameters(rewards, log_probs, entropies)

        if i_episode % config.ckpt_freq == 0:
            torch.save(agent.model.state_dict(), os.path.join(dir, '4.25-reinforce-'+str(i_episode)+'.pkl'))
        
        print("Episode: {}, reward: {}".format(i_episode, np.mean(rewards)))

    env.close()
    
