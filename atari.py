import multiprocessing
from multiprocessing import Process
import numpy as np
import time, os, shutil, random, pickle, gym
from cmath import inf
from cgp import *
import cgp
import matplotlib.pyplot as plt
from configuration import config
import networkx as nx
import warnings
from utils import extract_computational_subgraph
warnings.filterwarnings('ignore')

np.random.seed(config.seed)
random.seed(config.seed)



def func(idx, pop):
    '''
    idx: 进程号
    pop: 种群
    '''
    reward_pop = []
    env = gym.make('LunarLander-v2')
    # 遍历每个individual
    for p in pop:
        r_ind = 0
        # 每个individual测试5个episode
        for i in range(config.Epoch):
            s = env.reset()
            r_epoch = 0
            done = False
            while not done:
                action = p.eval(*s)
                action = np.exp(action)/np.exp(action).sum()
                action = np.argmax(action)
                # action = np.random.choice(4,p=action)
                s,r,done,_ = env.step(action)
                r_epoch += r
            r_ind += r_epoch
        r_ind /= config.Epoch
        p.fitness = r_ind
        reward_pop.append(r_ind)

    with open('./tmp/'+str(idx)+'.pkl','wb') as f:
        pickle.dump(reward_pop,f)
    # print(idx,' finished!')


pop = create_population(config.MU+config.LAMBDA,input_dim=8,out_dim=4)
best_f = -inf
best_ff = -inf
best_ind = None




total_agent = config.MU + config.LAMBDA

agent_p = int(total_agent/config.n_process)

for g in range(config.N_GEN):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp/')
    tick = time.time()
    process = []
    for i in range(config.n_process):
        process.append(Process(target=func, args=(i, pop[i*agent_p:(i+1)*agent_p])))

    [p.start() for p in process]
    [p.join() for p in process]


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
    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time:',time.time()-tick, pop[0].fitness)
    if pop[0].fitness > config.solved:
        with open('./results/best-100.pkl','wb') as f:
            pickle.dump(pop,f)
        break

env = gym.make('LunarLander-v2')
rr = 0
for i in range(100):
    r_e = 0
    done = False
    s = env.reset()
    while not done:
        action = pop[0].eval(*s)
        action = np.exp(action)/np.exp(action).sum()
        action = np.argmax(action)
        # action = np.random.choice(4,p=action)
        s,r,done,_ = env.step(action)
        r_e += r
    rr += r_e
    print(i,r_e)
print(rr/100)