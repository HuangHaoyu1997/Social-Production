import matplotlib.pyplot as plt
import numpy as np
from configuration import config
from utils import *
from env import Env
import pickle
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:10]

plt.ion()

env = Env()
env.reset()
total_coin = [[],[],[],[]]
agent_num = [[],[],[]]
agent_coin = [[],[],[],[]]
RSV = []
RH = [0.] # RH for Rate of Hire
data = []
if config.render:
    fig1 = plt.figure(1,(16,9))
    ax1 = fig1.add_subplot(231)
    ax2 = fig1.add_subplot(232)
    ax3 = fig1.add_subplot(233)
    ax4 = ax3.twinx()
    ax5 = fig1.add_subplot(234)
    ax6 = fig1.add_subplot(235)
    ax7 = fig1.add_subplot(236)

tick = time.time()
for t in range(config.T):
    coin_a, coin_v, coin_g, coin_t = total_value(env.agent_pool,env.market_V,env.gov)
    avg_coin_e, _ = avg_coin(env.agent_pool, env.E)
    avg_coin_w, _ = avg_coin(env.agent_pool, env.W)
    avg_coin_u, _ = avg_coin(env.agent_pool, env.U)
    avg_coin_t, _ = avg_coin(env.agent_pool, list(env.agent_pool.keys()))
    RateSurplusValue = exploit_ratio(env.agent_pool, env.E)
    if t%12==0:
        print('t,\tem,\tun,\two,\talive,\tagt_c,\tmkt_c,\tttl_c,\tavg_u,\tavg_e,\tavg_w')
    print('%d,\t%d,\t%d,\t%d,\t%d,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.2f'%\
        (t, len(env.E),len(env.U),len(env.W),alive_num(env.agent_pool),coin_a,coin_v,coin_t,avg_coin_u,avg_coin_e,avg_coin_w)
        )
    total_coin[0].append(coin_a)
    total_coin[1].append(coin_v)
    total_coin[2].append(coin_g)
    total_coin[3].append(coin_t)
    
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
    data_step = env.step(t, np.zeros((config.N)))
    data.extend(data_step)
    if t % 100 == 0:
        with open('./data/consume_data_'+run_time+'.pkl','wb') as f:
            pickle.dump(data, f)

    grid = grid_render(env.agent_pool,env.resource)
    #### Render
    if config.render:
        tick = time.time()
        
        ax1.cla()
        ax1.plot(agent_num[0])
        ax1.plot(agent_num[1])
        ax1.plot(agent_num[2])
        ax1.grid(); ax1.set_xlabel('T'); ax1.set_ylabel('Population')
        ax1.legend(['Unemployed','Employer','Worker'],loc=2)

        ax2.cla()
        ax2.plot(agent_coin[0])
        ax2.plot(agent_coin[1])
        ax2.plot(agent_coin[2])
        ax2.plot(agent_coin[3])
        ax2.grid(); ax2.set_xlabel('T'); ax2.set_ylabel('Coin')
        ax2.legend(['Unemployed','Employer','Worker','Total'],loc=2)


        ax3.cla()
        ax3.plot(RSV,'r')
        ax3.set_ylabel('Rate of Surplus Value',color='red')
        ax3.set_xlabel('T')
        ax3.grid()
        
        ax4.cla()
        ax4.plot(RH,'b')
        ax4.set_ylabel('Employment Rate',color='blue')
        # ax4.legend(['Rate of Surplus Value','Rate of Employment'],loc=2)
        # plt.legend(['single','total'])
        
        ax5.cla()
        ax5.plot(total_coin[0])
        ax5.plot(total_coin[1])
        ax5.plot(total_coin[2])
        ax5.plot(total_coin[3])
        ax5.grid(); ax5.set_xlabel('T'); ax5.set_ylabel('Coin')
        ax5.set_title('total coin')
        ax5.legend(['agent','market','government','Total'],loc=2)

        
        ax6.cla()
        ax6.imshow(grid)


        ax7.cla()
        p = nx.spring_layout(env.G)
        nx.draw(env.G, pos=p)
        plt.show()

        '''
        # 有问题,会影响figure(4)的显示
        # TODO 如何动态显示大规模nx.Graph？


        
        '''
        
        plt.pause(0.0001)
        print("tock = %.3f"%(time.time()-tick))

print('total time: %.3f,time per step:%.3f'%(time.time()-tick, (time.time()-tick)/config.T))