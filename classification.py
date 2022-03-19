
from cmath import inf
import numpy as np
from cgp import *
import cgp
import matplotlib.pyplot as plt
from configuration import config
import networkx as nx
import random
import pickle
import warnings
from utils import *
warnings.filterwarnings('ignore')

with open('./data/consume_data.pkl','rb') as f:
    data = pickle.load(f)

#         0    1              2           3             4          5 6
# data = [name,config.consume,coin_before,hungry_before,coin_after,V,hungry_after]
random.shuffle(data)
print(len(data))
print(data[0])
'''
consume = []
coin_before = []
coin_after = []
hungry_before = []
hungry_after = []
for d in data:
    consume.append(d[1])
    coin_before.append(d[2])
    hungry_before.append(d[3])
    coin_after.append(d[4])
    hungry_after.append(d[6])

'''

# np.random.seed(123)
# random.seed(123)

pop = create_population(config.MU+config.LAMBDA,input_dim=3,out_dim=2)
best_f = -inf
best_ff = -inf
best_ind = None


def eval(ind):
    loss = 0
    l = len(data[:1000])
    for d in data[:1000]:
        din = [d[1], d[2], d[3]]
        dout = [d[4], d[6]]
        out = ind.eval(*din)
        loss += 0.01*abs(out[0]-dout[0]) + float(out[1]!=dout[1])
        # print(abs(out[0]-dout[0]), float(out[1]!=dout[1]))
        
    # y_ = np.array(y_)
    # error = y_ - train_y
    return loss / l# (error!=0).sum()/120

def test(ind):
    loss = 0
    l = len(data[1000:])
    for d in data[1000:]:
        din = [d[1], d[2], d[3]]
        dout = [d[4], d[6]]
        out = ind.eval(*din)
        loss += 0.01*abs(out[0]-dout[0]) + float(out[1]!=dout[1])
        
    # y_ = np.array(y_)
    # error = y_ - train_y
    return loss/l # (error!=0).sum()/120

# 进化代际
acc_list = []
for gen in range(config.N_GEN):
    # 个体fitness eval
    for ind in pop:
        loss = eval(ind)
        ind.fitness = -loss
        if ind.fitness > best_f:
            best_ff = best_f
            best_ind = ind
            best_f = ind.fitness
            # print('best,',ind.fitness)
    # if abs(best_f-best_ff) <= 0.0001: prob *= 0.5

    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(gen,'\t',-pop[0].fitness,test(pop[0])) # ,avg_fit(pop)
    G = extract_computational_subgraph(pop[0])
    plt.clf(); plt.cla()
    pos = nx.circular_layout(G)
    node_labels = nx.get_node_attributes(G, 'func')
    # print(node_labels)
    nx.draw_networkx_labels(G, pos, labels=node_labels,font_size=20)  # 将func属性，显示在节点上
    edge_labels = nx.get_edge_attributes(G, 'weight') # 获取边的weight属性，
    # print(edge_labels) 
    nx.draw_networkx_edges(G, pos,)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20) # 将name属性，显示在边上

    plt.tight_layout()
    plt.savefig('./results/'+str(gen)+'.png')
    # plt.show()

    acc_list.append([-pop[0].fitness, test(pop[0])])

acc_list = np.array(acc_list)
plt.figure(1)
plt.plot(acc_list[:,0])
plt.plot(acc_list[:,1])
plt.xlabel('generation')
plt.ylabel('classification loss')
plt.legend(['train','test'])
plt.savefig('./img/iris_class.png')
plt.show()
# plt.show()

# test
acc_list = []
for ind in pop:
    loss = test(ind)
    acc_list.append(loss)
print(np.min(acc_list))



