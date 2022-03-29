import numpy as np
import matplotlib.pyplot as plt 
from typing import Dict, Generator, List, Optional, Set, Tuple, Type
import time
from configuration import config
config.seed = 123# int(time.time())
from cgp import *

def obj_fun(x):
    return x**2+2*x+1

def gen_data():
    y = []
    x = np.linspace(-3,1,200)
    for i in x:
        y.append(obj_fun(i))
    return x, np.array(y)

def NRMSE(y:np.ndarray, y_pred:np.ndarray):
    '''
    normalized root-mean-square error (NRMSE)
    用真实回归数据的标准差进行归一化
    '''
    nrMSE = np.sqrt((y - y_pred).mean()) / y.std()
    return 1/(1+nrMSE)

x,y = gen_data()

pop = create_population(config.MU+config.LAMBDA, input_dim=1, out_dim=1)

def cgp_regressor(pop, data, N_gen, terminate_fit, fitness_function=NRMSE):
    for i in range(N_gen):
        for ind in pop:
            y_pred = []
            for sample in data[0]:
                y_pred.append(ind.eval(sample))
            fit = fitness_function(data[1], y_pred)
            ind.fitness = fit
        pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
        print(i,pop[0].fitness)
        if pop[0].fitness >= terminate_fit:
            break
    return pop
cgp_regressor(pop, [x,y], config.N_GEN, 1.0)
#plt.figure()
#plt.plot(x,y)
#plt.show()
# print(x,y)
