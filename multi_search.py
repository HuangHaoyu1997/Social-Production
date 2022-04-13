'''
多进程CGP算法训练Social Production policy
'''
from multiprocessing import Process
import numpy as np
import time, os, shutil, random, pickle, gym
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

np.random.seed(config.seed)
random.seed(config.seed)


def func(idx, pop):
    '''
    子进程所执行的函数
    idx: 进程号
    pop: 种群
    '''
    reward_pop = []
    env = Env()
    tick = time.time()
    # 遍历每个individual
    for p in pop:
        r_ind = 0
        # 每个individual测试5个episode
        
        for i in range(config.Epoch):
            info = env.reset() # ; s = info_parser(info)
            r_epoch = 0
            done = False
            while not done:
                # action = p.eval(*s)
                # action = np.clip(action,1.,100.)
                action = uniform(0,100)
                info, r, done = env.step(action)
                # s = info_parser(info)
                r_epoch += r
            r_ind += r_epoch
        r_ind /= config.Epoch
        p.fitness = r_ind
        reward_pop.append(r_ind)
    print(idx, (time.time()-tick) / (len(pop)*config.Epoch))
    '''
    with open('./tmp/'+str(idx)+'.pkl','wb') as f:
        pickle.dump(reward_pop,f)
    
    '''
    # print(idx,' finished!')

pop = create_population(config.MU+config.LAMBDA,input_dim=22,out_dim=1)
best_f = -inf
best_ff = -inf
best_ind = None

total_agent = config.MU + config.LAMBDA
agent_p = int(total_agent/config.n_process) # 平均每个进程分到的agent数量

# 开始搜索
for g in range(config.N_GEN):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp/')
    
    # 运行1代总时间
    tick = time.time()
    process = []
    
    for i in range(config.n_process):
        process.append(Process(target=func, args=(i, pop[i*agent_p:(i+1)*agent_p])))

    [p.start() for p in process]
    [p.join() for p in process]
    '''
    fitness = []
    for i in range(config.n_process):
        with open('./tmp/'+str(i)+'.pkl','rb') as f:
            data = pickle.load(f)
            fitness.extend(data)
    for f,p in zip(fitness,pop):
        p.fitness = f

    fitness = np.array(fitness)
    idx = fitness.argsort()[::-1][0:config.MU]
    shutil.rmtree('./tmp/',True)
    '''
    
    # pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time for one generation:', time.time()-tick, pop[0].fitness)
    #if pop[0].fitness > config.solved:
    if g % 10 == 9:
        with open('./results/SP-'+str(g)+'.pkl','wb') as f:
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