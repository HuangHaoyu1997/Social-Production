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
agent_num = [[],[],[],[]]
agent_coin = [[],[],[],[]]
w = [[],[]]
RSV = []
RH = [0.] # RH for Rate of Hire
JoblossRate = []
data = []


tick = time.time()
for t in range(config.T):
    
    
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
    agent_num[3].append(alive_num(env.agent_pool))

    JoblossRate.append(agent_num[0][-1] / agent_num[3][-1])

    agent_coin[0].append(avg_coin_u)
    agent_coin[1].append(avg_coin_e)
    agent_coin[2].append(avg_coin_w)
    agent_coin[3].append(avg_coin_t)

    w[0].append(env.w1)
    w[1].append(env.w2)

    RSV.append(RateSurplusValue)
    if t > 0: RH.append(len(env.W)/len(env.E))
    
    #### Data storage
    data_step = env.step(t, ) # np.zeros((config.N1))

    if t >= 100 and t < 100+config.event_duration: # t%100 == 99:
        env.event_simulator('GreatDepression')

    data.extend(data_step)
    if t % 100 == 0:
        with open('./data/consume_data_'+run_time+'.pkl','wb') as f:
            pickle.dump(data, f)

    grid = grid_render(env.agent_pool,env.resource)
    #### Render
    if t%100==99 and config.render:
        tick = time.time()
        fig1 = plt.figure(1,(25,10))
        ax1 = fig1.add_subplot(241) # 各部人口变化趋势图
        ax2 = fig1.add_subplot(242) # 各部平均财富图
        ax3 = fig1.add_subplot(243) # 剥削率
        ax4 = ax3.twinx()           # 雇佣率
        ax5 = fig1.add_subplot(244) # 各子系统总财富变化图
        ax6 = fig1.add_subplot(245) # Grid可视化
        ax7 = fig1.add_subplot(246) # Graph可视化
        ax8 = fig1.add_subplot(247) # 最低工资、最高工资变化趋势图
        ax9 = fig1.add_subplot(248) # 失业率
        
        ax1.cla()
        ax1.plot(agent_num[0])
        ax1.plot(agent_num[1])
        ax1.plot(agent_num[2])
        ax1.plot(agent_num[3])
        ax1.grid(); ax1.set_xlabel('T'); ax1.set_ylabel('Population')
        ax1.legend(['Unemployed','Employer','Worker','Total'],loc=2)

        ax2.cla()
        ax2.plot(agent_coin[0])
        ax2.plot(agent_coin[1])
        ax2.plot(agent_coin[2])
        ax2.plot(agent_coin[3])
        ax2.grid(); ax2.set_xlabel('T'); ax2.set_ylabel('Coin')
        ax2.set_yscale('log')
        ax2.set_title('avg_coin')
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
        ax5.set_yscale('log')
        ax5.set_title('total coin')
        ax5.legend(['agent','market','government','Total'],loc=2)

        
        ax6.cla()
        ax6.imshow(grid)

        ax7.cla()
        ax7.plot(w[0])
        ax7.plot(w[1])
        ax7.grid(); ax7.set_xlabel('T'); ax7.set_ylabel('Salary')
        ax7.set_title('Top and bottom salary')
        ax7.legend(['bottom','top'],loc=2)

        

        ax8.cla()
        ax8.plot(JoblossRate)
        ax8.grid(); ax8.set_xlabel('T'); ax8.set_ylabel('Jobless Rate')
        ax8.set_title('Jobless Rate')

        ax9.cla()
        p = nx.spring_layout(env.G)
        nx.draw(env.G, pos=p)
        # plt.show()
        '''
        # 有问题,会影响figure(4)的显示
        # TODO 如何动态显示大规模nx.Graph?
        '''
        
        plt.pause(0.0001)
        # print("tock = %.3f"%(time.time()-tick))
        plt.savefig('./results/'+str(t)+'.png')
        plt.close()
time.sleep(7200)
print('total time: %.3f,time per step:%.3f'%(time.time()-tick, (time.time()-tick)/config.T))