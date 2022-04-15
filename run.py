'''
Social Production的运行示例
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from configuration import config
from utils import *
from env import Env
import pickle, os, time
from collections import Counter
matplotlib.use('pdf')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 运行时间e.g.'2022-04-07-15-10-12'
run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
plt.ion()

env = Env()

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
        if done:
            env.render()
        '''
        data.extend(data_step)
        if env.t % 100 == 0:
            with open('./data/consume_data_'+run_time+'.pkl','wb') as f:
                pickle.dump(data, f)
        '''
        # 收集数据
        '''
        
        '''
        
        #### Render
        
    print('total time: %.3f,time per step:%.3f'%(time.time()-tick, (time.time()-tick)/config.T), Counter(env.death_log)[1],Counter(env.death_log)[2],len(env.death_log))
print('done')
time.sleep(7200)
