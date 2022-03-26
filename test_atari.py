'''
测试CGP个体在Atari游戏上的performance
'''

import gym
import pickle
import numpy as np

with open('./results/best-100.pkl','rb') as f:
    pop = pickle.load(f)

env = gym.make('LunarLander-v2')

for i in range(100):
    rr = 0
    s = env.reset()
    done = False
    while not done:
        action = pop[0].eval(*s)
        action = np.exp(action)/np.exp(action).sum()
        action = np.argmax(action)
        s,r,done,_ = env.step(action)
        env.render()
        rr += r
    print(i,rr)
