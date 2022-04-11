'''
单进程CGP算法训练Social Production policy
'''
import numpy as np
import time, os, shutil, random, pickle
from cmath import inf

# ['coin_a', 
                # 'coin_v', 
                # 'coin_g', 
                # 'coin_t', 
                # 'avg_coin_e', 
                # 'std_coin_e', 
                # 'avg_coin_w', 
                # 'std_coin_w', 
                # 'avg_coin_u', 
                # 'std_coin_u', 
                # 'avg_coin_t', 
                # 'std_coin_t', 
                # 'Upop', 
                # 'Wpop', 
                # 'Epop', 
                # 'Tpop', 
                # 'RJ', 
                # 'RSV', 
                # 'RH', 
                # 'w1', 
                # 'w2',
                # 'avg_age',]
                

from cgp import *
import matplotlib.pyplot as plt
from configuration import config
import networkx as nx
from env import Env
import warnings

from utils import info_parser
warnings.filterwarnings('ignore')

run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:10]
np.random.seed(config.seed)
random.seed(config.seed)


def func(pop):
    '''
    CGP所执行的函数
    pop: 种群
    '''
    env = Env()
    # 遍历每个individual
    for p in pop:
        r_ind = 0
        # 每个individual测试5个episode
        for i in range(config.Epoch):
            info = env.reset(); s = info_parser(info)
            r_epoch = 0
            done = False
            while not done:
                action = p.eval(*s)
                action = np.clip(action,1.,100.)
                info, r, done = env.step(action)
                s = info_parser(info)
                r_epoch += r
            r_ind += r_epoch
        r_ind /= config.Epoch
        p.fitness = r_ind
    return pop

pop = create_population(config.MU+config.LAMBDA,input_dim=22,out_dim=1)
best_f = -inf
best_ff = -inf
best_ind = None


# 开始搜索
for g in range(config.N_GEN):
    tick = time.time()
    pop = func(pop)
    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time:',time.time()-tick, pop[0].fitness)
    #if pop[0].fitness > config.solved:
    if g % 1 == 0:
        with open('./results/'+run_time+'-SP-'+str(g)+'.pkl','wb') as f:
            pickle.dump(pop,f)

env = Env()
rr = 0
for i in range(100):
    r_e = 0
    done = False
    s = info_parser(env.reset())
    while not done:
        action = pop[0].eval(*s)
        action = np.min(np.max(action, 0), 200)
        # action = np.random.choice(4,p=action)
        info, r, done = env.step(action)
        s = info_parser(info)
        r_e += r
    rr += r_e
    print(i, r_e)
print(rr/100)