'''
Social Production的运行示例
'''
import matplotlib, pickle, os, time
import matplotlib.pyplot as plt
import numpy as np
from configuration import config
from utils import *
from env import Env
from collections import Counter
matplotlib.use('pdf') # 不显示图片,直接保存pdf

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
    
    count = 0
    while not done:
        if count <= 100:
            action = 30.0
        else:
            action = 80.0
        info, reward, done = env.step( action ) # np.zeros((config.N1))
        data.append(info_parser(info))

        for event_point in config.event_point:
            if abs(env.t - event_point) <= config.event_duration: # t%100 == 99:
                env.event_simulator('GreatDepression')
        count += 1
        run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
        if done: env.render('./results/exp1/'+run_time+'_'+'test'+'.pdf')
        '''
        data.extend(data_step)
        if env.t % 100 == 0:
            with open('./data/consume_data_'+run_time+'.pkl','wb') as f:
                pickle.dump(data, f)
        '''
    with open('./data/tSNE-simulation.pkl','wb') as f:
        pickle.dump(data, f)
    print('total time: %.3f,time per step:%.3f'%(time.time()-tick, (time.time()-tick)/config.T), Counter(env.death_log)[1],Counter(env.death_log)[2],len(env.death_log))
print('done')
time.sleep(7200)

