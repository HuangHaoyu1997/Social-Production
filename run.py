import matplotlib.pyplot as plt
import numpy as np
from configuration import config
from utils import *
from env import Env
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.ion()

env = Env()
env.reset()
agent_num = [[],[],[]]
agent_coin = [[],[],[],[]]
RSV = []
RH = [0.] # RH for Rate of Hire
data = []
for t in range(config.T):
    coin_a, coin_v, coin_t = total_value(env.agent_pool,env.market_V)
    avg_coin_e, _ = avg_coin(env.agent_pool, env.E)
    avg_coin_w, _ = avg_coin(env.agent_pool, env.W)
    avg_coin_u, _ = avg_coin(env.agent_pool, env.U)
    avg_coin_t, _ = avg_coin(env.agent_pool, list(env.agent_pool.keys()))
    RateSurplusValue = exploit_ratio(env.agent_pool, env.E)
    print('%d,\t%d,\t%d,\t%d,\t%d,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.2f'%\
        (t, len(env.E),len(env.U),len(env.W),alive_num(env.agent_pool),coin_a,coin_v,coin_t,avg_coin_u,avg_coin_e,avg_coin_w)
        )
    agent_num[0].append(len(env.U))
    agent_num[1].append(len(env.E))
    agent_num[2].append(len(env.W))

    agent_coin[0].append(avg_coin_u)
    agent_coin[1].append(avg_coin_e)
    agent_coin[2].append(avg_coin_w)
    agent_coin[3].append(avg_coin_t)

    RSV.append(RateSurplusValue)
    if t > 0: RH.append(len(env.W)/len(env.E))
    
    #### Data storage
    data_step = env.step(np.zeros((config.N)))
    data.extend(data_step)
    if t % 100 == 0:
        with open('./data/consume_data.pkl','wb') as f:
            pickle.dump(data, f)

    grid = grid_render(env.agent_pool,env.resource)
    #### Render
    if config.render:
        plt.clf()
        plt.figure(1)
        plt.cla()
        plt.plot(agent_num[0])
        plt.plot(agent_num[1])
        plt.plot(agent_num[2])
        plt.grid(); plt.xlabel('T'); plt.ylabel('Population')
        plt.legend(['Unemployed','Employer','Worker'],loc=2)

        plt.figure(2)
        plt.cla()
        plt.plot(agent_coin[0])
        plt.plot(agent_coin[1])
        plt.plot(agent_coin[2])
        plt.plot(agent_coin[3])
        plt.grid(); plt.xlabel('T'); plt.ylabel('Coin')
        plt.legend(['Unemployed','Employer','Worker','Total'],loc=2)

        fig3 = plt.figure(3)
        ax1 = fig3.add_subplot(111)
        # fig3.clf()
        ax1.cla()
        ax1.plot(RSV,'r')
        ax1.set_ylabel('Rate of Surplus Value',color='red')
        ax1.set_xlabel('T')
        ax1.grid()
        
        ax2 = ax1.twinx() 
        ax2.cla()
        ax2.plot(RH,'b')
        ax2.set_ylabel('Employment Rate',color='blue')
        fig3.legend(['Rate of Surplus Value','Rate of Employment'],loc=2)
        # plt.legend(['single','total'])
        
        '''
        fig4 = plt.figure(4)
        ax1 = fig4.add_subplot(111)
        ax1.cla()
        ax1.imshow(grid)

        '''
        

        '''
        # 有问题,会影响figure(3)的显示
        # TODO 如何动态显示大规模nx.Graph？


        fig4 = plt.figure(4)
        ax = fig4.add_subplot(111)
        ax.cla()
        p = nx.spring_layout(env.G)
        nx.draw(env.G, pos=p)
        plt.show()
        '''
        
        plt.pause(0.0001)

