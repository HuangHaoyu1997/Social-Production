import numpy as np
import matplotlib.pyplot as plt 
from typing import Dict, Generator, List, Optional, Set, Tuple, Type
import time
from configuration import config
config.seed = 123 # int(time.time())
from cgp import *
from postprocessing import *

def obj_fun(x):
    return x*x + 2*x + 1


def gen_data():
    y = []
    x = np.linspace(-10,10,500) # 数据量太少、定义域太窄，会搜索到错误解，但奇怪的是错误解依然满足loss=0
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
plt.figure()
plt.plot(x,y)
plt.show()

pop = create_population(config.MU+config.LAMBDA, input_dim=1, out_dim=1)

def cgp_regressor(pop, data, N_gen, terminate_fit, fitness_function=NRMSE):
    '''
    CGP数据拟合学习器
    pop: popuplation
    data: 拟合数据
    N_gen: 进化代数
    terminate_fit: 停止搜索条件, ≥terminate_fit即停止
    fitness_function: 适应度计算函数

    return: 进化后的pop, best_fit
    '''
    for i in range(N_gen):
        
        for ind in pop:

            y_pred = []
            for sample in data[0]:
                # print(sample)
                y_pred.append(ind.eval(*[sample]))
            fit = fitness_function(data[1], y_pred)
            ind.fitness = fit
        pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
        print(i,pop[0].fitness)
        
        if pop[0].fitness >= terminate_fit:
            break

    return pop, pop[0].fitness


pop, best_fit = cgp_regressor(pop, [x,y], config.N_GEN, 1.0)

g = extract_computational_subgraph(pop[0])
formula = simplify(g, ['x'])
formula = round_expr(formula, config.PP_FORMULA_NUM_DIGITS)
print(pop[0].fitness, formula)
visualize(g, "./results/xx+2x+1.jpg", input_names=['x'], operator_map=DEFAULT_SYMBOLIC_FUNCTION_MAP)

# print(x,y)
