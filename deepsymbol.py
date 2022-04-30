from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x = F.softmax(x, dim=-1)
        idx = torch.argmax(x,dim=-1).unsqueeze_(0)
        print(idx.shape, x.shape)
        return x

model = Model(inpt_dim=4, hid_dim=12, dict_dim=5)
sym_mat = model(torch.tensor(s))
# print(sym_mat[1,:,:])

