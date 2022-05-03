import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from function import Function

import gym

env = gym.make('CartPole-v1')
s = env.reset()

class Model(nn.Module):
    def __init__(self, inpt_dim, hid_dim, dict_dim, ):
        super(Model, self).__init__()
        self.inpt_dim = inpt_dim 
        self.hid_dim = hid_dim
        self.dict_dim = dict_dim
        self.fc1 = nn.Linear(inpt_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dict_dim*inpt_dim*inpt_dim)

    def forward(self, x, ):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(self.inpt_dim, self.inpt_dim, self.dict_dim)
        x = F.softmax(x, dim=-1) # x.shape=(4,4,5)
        
        return x

class DeepSymbol():
    def __init__(self) -> None:
        self.model = Model(inpt_dim=4, hid_dim=12, dict_dim=5)

    def sym_mat(self, state):
        '''
        generate the symbol matrix for observation vector
        '''
        matrix_prob = self.model(torch.tensor(state))
        
        # find the max prob symbol for each position in matrix
        idx = torch.argmax(matrix_prob, dim=-1)
        
        # calculate the joint log prob for all symbol selected
        log_prob = 0
        for i in range(idx.size()[0]):
            for j in range(idx.size()[1]):
                log_prob += matrix_prob[i, j, idx[i,j]].log()
        print(log_prob)

        # calculate upper bound of entropy for joint symbol categorical distribution
        entropies = 0
        for i in range(idx.size()[0]):
            for j in range(idx.size()[1]):
                p = matrix_prob[i, j]
                dist = Categorical(p)
                entropy = dist.entropy()
                entropies += entropy
        print(entropies)
        # print(idx.shape, x.shape)
        print(idx)
    
    def execute(self, idx):
        

def torchInv(x):
    return torch.pow(x, -1)
def torchConst(x):
    return torch.pow(x, 0)
func_set = [
    Function(torch.add, 2, 'torchAdd'),
    Function(torch.sub, 2, 'torchSub'),
    Function(torch.mul, 2, 'torchMul'),
    Function(torch.div, 2, 'torchDiv'),
    Function(torch.max, 2, 'torchMax'),
    Function(torch.min, 2, 'torchMin'),

    Function(torch.log, 1, 'torchLog'),
    Function(torch.sin, 1, 'torchSin'),
    Function(torch.cos, 1, 'torchCos'),
    Function(torch.exp, 1, 'torchExp'),
    Function(torch.neg, 1, 'torchNeg'),
    Function(torch.abs, 1, 'torchAbs'),
    Function(torch.square, 1, 'torchX^2'),
    Function(torch.sqrt, 1, 'torchSqrt'),
    Function(torch.sign, 1, 'torchSgn'),
    Function(torch.relu, 1, 'torchRelu'),
    Function(torchInv, 1, 'torchInv'),
    Function(torchConst, 1, 'torchConst')

]

ds = DeepSymbol()
ds.sym_mat(s)
