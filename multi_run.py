'''
Social Production多环境并行化运行示例
基于gym vector API
'''

import matplotlib, gym
import matplotlib.pyplot as plt
import numpy as np
from configuration import config
from utils import *
from env import Env
import pickle, os, time, random
from collections import Counter
matplotlib.use('pdf') # 不显示图片,直接保存pdf

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 运行时间e.g.'2022-04-07-15-10-12'
run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
plt.ion()

env = Env()
class DictEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "position": gym.spaces.Box(-1., 1., (3,), np.float32),
        "velocity": gym.spaces.Box(-1., 1., (2,), np.float32)
    })
    action_space = gym.spaces.Dict({
        "fire": gym.spaces.Discrete(2),
        "jump": gym.spaces.Discrete(2),
        "acceleration": gym.spaces.Box(-1., 1., (2,), np.float32)
    })
    def reset(self):
        self.flag = False
        return self.observation_space.sample()

    def step(self, action):
        observation = self.observation_space.sample()
        if not self.flag:
            done = True if random.random()<0.2 else False
        else: done = True
        if done: self.flag = True
        return (observation, 0., done, {})

env_list = [
        lambda: DictEnv()
    ]*5
envs = gym.vector.SyncVectorEnv(env_list)
# envs = gym.vector.AsyncVectorEnv(env_list)

tick = time.time()
envs.reset()
    
for i in range(10000):
    s, r, done, _ = envs.step(envs.action_space.sample())
    # print(done)
print((time.time()-tick)/10000*1000)

'''
for i in range(10):
    tick = time.time()
    done = False
    config.seed = i
    env.reset()
    data = []
    
    while not done:
        info, reward, done = env.step( uniform(0,100) ) # np.zeros((config.N1))
        for event_point in config.event_point:
            if abs(env.t - event_point) <= config.event_duration: # t%100 == 99:
                env.event_simulator('GreatDepression')
        if done: env.render()
        #### Render
        
    print('total time: %.3f,time per step:%.3f'%(time.time()-tick, (time.time()-tick)/config.T), Counter(env.death_log)[1],Counter(env.death_log)[2],len(env.death_log))
print('done')
time.sleep(7200)
'''
