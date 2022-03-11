from tkinter import W
import numpy as np
from config import config
from agent import agent
import random
import copy

class env:
    def __init__(self,param) -> None:
        self.param = param
        
    
    def reset(self,):
        self.t = 0 # tick
        self.agent_num = self.param.N # 智能体数量
        self.agent_pool = self.add_agent(self.agent_num) # 添加智能体
        self.market_V = self.param.V # 市场价值
        self.E, self.W, self.U = self.working_state() # 更新维护智能体状态

        return [self.E, self.W ,self.U, self.market_V]

    def add_agent(self, N):
        pool = {}
        for i in range(N):
            param = config()
            param.name = str(i)
            pool[param.name] = agent(param)
        return pool
    
    def working_state(self,):
        '''
        统计当前所有agent工作状态
        E:Employer
        W:Worker
        U:Unemployed
        '''
        
        E, W, U = [], [], []
        for name in self.agent_pool:
            if self.agent_pool[name].work == 0:
                # 失业者
                U.append(name)
            elif self.agent_pool[name].work == 1:
                # 雇主是自己，自己是资本家
                E.append(name)
            elif self.agent_pool[name].work == 2:
                # 雇主不是自己，自己是工人
                W.append(name)
        return E, W, U
    
    def step(self,): # 单步执行函数
        
        self.hire()
        self.exploit()
        self.pay()
        self.consume()

    def hire(self,):
        U = copy.deepcopy(self.U)
        E = copy.deepcopy(self.E)
        random.shuffle(U)
        UE = U+E
        random.shuffle(UE)
        for u in U: # 遍历所有失业者
            
            if self.agent_pool[u].work!=0: continue
            # u的潜在雇佣者的货币统计
            potential_e = [self.agent_pool[name].coin for name in UE]
            # 按货币量多少概率决定u的雇主
            prob = np.array(potential_e) / np.array(potential_e).sum()
            e = UE[np.random.choice(np.arange(len(prob)),p=prob)]
            # print(len(potential_e),len(prob),e,len(UE))
            
            # config.avg_coin = avg_coin(self.agent_pool,self.W+self.U)[0] # 更新平均工资
            if self.agent_pool[e].coin >= config.avg_coin:
                # 给u设置雇主e,修改工作状态
                self.agent_pool[u].employer = e
                self.agent_pool[u].work = 2 # 2 for worker
                # 给雇主e添加员工u,修改工作状态
                self.agent_pool[e].employer = e
                self.agent_pool[e].hire.append(u)
                self.agent_pool[e].work = 1 # 1 for employer
                u_idx = UE.index(u); UE.pop(u_idx)
                # print(e,self.agent_pool[e].work)
                # e_idx = UE.index(e)
        
        self.E, self.W, self.U = self.working_state() # 更新维护智能体状态


    def exploit(self,):
        for name in self.agent_pool:
            if self.agent_pool[name].work == 0: continue
            a = self.agent_pool[name]
            a_e = a.employer
            w = np.random.randint(low=0,high=int(config.danjia*self.market_V)) # 1000
            self.market_V -= w
            self.agent_pool[a_e].coin += w



    def pay(self,):
        # 遍历E中的每个employer，为employer.hire中的worker发工资
        for employer in self.E:
            capital = self.agent_pool[employer].coin # employer现有资本
            worker_list = self.agent_pool[employer].hire # 雇佣名单
            random.shuffle(worker_list)
            for worker in worker_list:
                # 在最低工资和最高工资之间发工资
                w = np.random.randint(self.param.w1,self.param.w2)
                if capital >= w:
                    self.agent_pool[worker].coin += w
                    capital -= w
                # 资本量不足以开出现有工资
                elif capital < w and capital > self.param.w1:
                    w = np.random.randint(self.param.w1,capital) # 降薪发工资
                    self.agent_pool[worker].coin += w
                    capital -= w
                elif capital <= self.param.w1:
                    self.agent_pool[worker].coin += capital # 破产发工资
                    capital = 0
                    break
            if capital <= 0:
                self.broken(employer) # 破产
                self.E, self.W, self.U = self.working_state() # 更新工作状态
        return None

    def broken(self,employer):
        '''
        雇佣者及其雇工都失业,加入U集合,从E和W集合中删除对应agent
        被雇者的雇主设置为0,修改其工作状态
        雇佣者的雇佣名单清空,修改其工作状态
        
        '''
        self.agent_pool[employer].work = 0 # 失业
        for worker in self.agent_pool[employer].hire:
            self.agent_pool[worker].work = 0 # 失业
            self.agent_pool[worker].employer = None
        self.agent_pool[employer].hire = []
        
        return None

    def fire(self):
        '''
        解雇函数

        '''
        for employer in self.E:
            # 该资本家的货币量
            capital = self.agent_pool[employer]
            # 雇工名单
            worker_list = self.agent_pool[employer].hire
            # 最大工人数量=货币量除以平均工资
            max_num = np.floor(2*capital / (self.param.w1+self.param.w2))
            # 工人数量
            num_worker = len(worker_list)
            fire_num = np.max(num_worker - max_num,0) # 解雇数量
            if fire_num > 0:
                fire_list = random.sample(self.agent_pool[employer].hire,fire_num) # 随机解雇
                for worker in fire_list:
                    # 改变工作状态
                    self.agent_pool[worker].work = 0
                    self.agent_pool[worker].employer = None
                    # 修改雇工名单
                    fid = self.agent_pool[employer].hire.index(worker)
                    self.agent_pool[employer].hire.pop(fid)

    def consume(self,):
        # 消费，随机量m∈[0,m_a]
        # agent的钱减少m，市场价值增加m
        V_value = 0
        for name in self.agent_pool:
            ma = self.agent_pool[name].coin
            if ma > 0: m = np.random.randint(0,ma)
            else: m = 0
            self.agent_pool[name].coin -= m
            V_value += m
        
        self.market_V += V_value
        return None

def total_value(agent_pool, V):
    # 系统总货币量
    M = 0
    for name in agent_pool:
        M += agent_pool[name].coin
    return M+V

def avg_coin(agent_pool, agent):
    if len(agent)==0: return 0., 0.
    count = []
    for name in agent:
        count.append(agent_pool[name].coin)
    count = np.array(count)
    return np.round(count.mean(),2), np.round(count.std(),2)


env = env(config)
env.reset()
for t in range(config.T):

    print(t, len(env.E),len(env.U),len(env.W),
        env.market_V,
        total_value(env.agent_pool,env.market_V),
        avg_coin(env.agent_pool,env.E),
        avg_coin(env.agent_pool,env.W))
    env.step()


# env.add_agent()
