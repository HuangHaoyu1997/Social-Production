from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

env = gym.make('CartPole-v1')
s = env.reset()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,16)

    def forward(self, x, ):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

model = Model()
print(model(torch.tensor(s)))

