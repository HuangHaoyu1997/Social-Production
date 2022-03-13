from tkinter.messagebox import NO
import numpy as np
from config import config
from agent import agent
import random
import math
import copy
from utils import *

random.seed(config.seed)
np.random.seed(config.seed)

class Env:
    def __init__(self) -> None:
        pass
        
    
    def reset(self,):
        self.t = 0 # tick
        self.agent_num = config.N # 智能体数量
        self.agent_pool = self.add_agent(self.agent_num) # 添加智能体
        self.market_V = config.V # 市场价值
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
            if self.agent_pool[name].alive:
                if self.agent_pool[name].work == 0:# 失业者
                    U.append(name)
                elif self.agent_pool[name].work == 1:# 雇主是自己，资本家
                    E.append(name)
                elif self.agent_pool[name].work == 2:# 雇主是别人，自己是工人
                    W.append(name)
        return E, W, U
    
    def step(self,action): # 单步执行函数
        if len(action)!=self.agent_num:
            raise Exception('动作空间必须与智能体数量相同')

        agent_list = list(self.agent_pool.keys())
        random.shuffle(agent_list)
        
        for name in agent_list:
            # 必须活着
            if self.agent_pool[name].alive is not True: continue
            self.hire(name)
            self.exploit(name)
            self.pay(name)
            self.consume(name)
        
        self.E, self.W, self.U = self.working_state()


    def hire(self, name):
        
        work = self.agent_pool[name].work
        
        # 【下一行能不能删去，安全吗？】
        self.E, self.W, self.U = self.working_state() # 可能耗费时间，确保都是活着的
        if work == 0: # 失业
            U = copy.deepcopy(self.U); E = copy.deepcopy(self.E); random.shuffle(U)
            UE = U+E; random.shuffle(UE)
            # name的潜在雇佣者的货币统计
            potential_e = [self.agent_pool[h].coin for h in UE]
            # 按货币量多少概率决定u的雇主
            prob = np.array(potential_e) / np.array(potential_e).sum()
            e = UE[np.random.choice(np.arange(len(prob)),p=prob)]
            if config.avg_update: 
                config.avg_coin = avg_coin(self.agent_pool,self.W+self.U)[0] # 更新平均工资
            if self.agent_pool[e].coin >= config.avg_coin and name!=e:
                # 给name设置雇主e,修改工作状态
                self.agent_pool[name].employer = e
                self.agent_pool[name].work = 2 # 2 for worker
                self.W.append(name)
                u_idx = self.U.index(name); self.U.pop(u_idx)
                # 给雇主e添加员工u,修改工作状态
                self.agent_pool[e].employer = e
                self.agent_pool[e].hire.append(name)
                self.agent_pool[e].work = 1 # 1 for employer
                if e not in self.E: self.E.append(e)
        elif work == 1: # 雇佣者
            pass
        elif work == 2: # 被雇佣者
            pass
        
        # 【这一行能不能删去，安全吗？】
        self.E, self.W, self.U = self.working_state()

    def exploit(self, name):
        work = self.agent_pool[name].work
        
        if work == 0: return None # 失业
        elif work == 1: employer = name # 雇主
        elif work == 2: employer = self.agent_pool[name].employer # 被雇佣

        w = np.random.randint(low=0,high=int(config.danjia*self.market_V)) # 1000
        self.market_V -= w
        self.agent_pool[employer].coin += w


    def pay_for_single(self, name):

        # 工作状态


        capital = self.agent_pool[name].coin # employer现有资本
        worker_list = self.agent_pool[name].hire # 雇佣名单
        random.shuffle(worker_list)
        for worker in worker_list:
            # 在最低工资和最高工资之间发工资
            w = np.random.randint(config.w1, config.w2)
            if capital >= w:
                self.agent_pool[worker].coin += w
                capital -= w
            # 资本量不足以开出现有工资
            elif capital < w and capital > config.w1:
                w = np.random.randint(config.w1, capital) # 降薪发工资
                self.agent_pool[worker].coin += w
                capital -= w
            # 连最低工资也开不出了
            elif capital <= config.w1:
                self.agent_pool[worker].coin += capital # 破产发工资
                capital = 0
                break
        
        if capital <= 0:
            self.broken(name) # 破产
            self.E, self.W, self.U = self.working_state() # 更新工作状态


    def pay(self, name):
        '''为agent_pool[name].hire中的worker发工资'''
        if self.agent_pool[name].work != 1: return None
        
        worker_list = self.agent_pool[name].hire # 雇佣名单
        random.shuffle(worker_list)
        for worker in worker_list:
            capital = self.agent_pool[name].coin # employer现有资本
            # 在最低工资和最高工资之间随机发工资
            w = np.random.randint(config.w1, config.w2)
            if capital >= w:
                self.agent_pool[worker].coin += w
                capital -= w
                self.agent_pool[name].coin = capital  ## 更新employer
            # 资本量不足以开出现有工资
            elif capital < w and capital > config.w1:
                w = np.random.randint(config.w1, capital) # 降薪发工资
                self.agent_pool[worker].coin += w
                capital -= w
                self.agent_pool[name].coin = capital  ## 更新employer
            elif capital <= config.w1:
                self.agent_pool[worker].coin += capital # 破产发工资
                capital = 0
                self.agent_pool[name].coin = capital  ## 更新employer
                break
        self.agent_pool[name].coin = capital # 【这一行有没有必要】
        if self.agent_pool[name].coin <= 0:
            self.broken(name) # 破产
        self.E, self.W, self.U = self.working_state() # 更新工作状态
        return None

    def broken(self,employer):
        '''
        雇佣者及其雇工都失业,加入U集合,从E和W集合中删除对应agent
        被雇者的雇主设置为0,修改其工作状态
        雇佣者的雇佣名单清空,修改其工作状态
        
        '''
        assert self.agent_pool[employer].work == 1
        if config.Verbose: print('%s has broken!'%employer)
        self.agent_pool[employer].work = 0 # 失业
        for worker in self.agent_pool[employer].hire:
            self.agent_pool[worker].work = 0 # 失业
            self.agent_pool[worker].employer = None
        self.agent_pool[employer].hire = []
        self.agent_pool[employer].employer = None
        return None

    def fire(self, name):
        '''
        解雇
        '''
        if self.agent_pool[name].work != 1: return None

        self.E, self.W, self.U = self.working_state() # 更新维护智能体状态
        
        # 该资本家的货币量
        capital = self.agent_pool[name].coin
        # 雇工名单
        worker_list = self.agent_pool[name].hire
        # 最大工人数量=货币量/平均工资
        config.avg_coin = avg_coin(self.agent_pool, self.W+self.U)[0]
        max_num = np.floor(capital / config.avg_coin)
        # 工人数量
        num_worker = len(worker_list)
        fire_num = int(max(num_worker-max_num, 0)) # 解雇数量
        if fire_num > 0:
            fire_list = random.sample(self.agent_pool[name].hire, fire_num) # 随机解雇
            for worker in fire_list:
                # 改变工作状态
                assert worker in self.agent_pool[name].hire
                self.agent_pool[worker].work = 0
                self.agent_pool[worker].employer = None
                # 修改雇工名单
                fid = self.agent_pool[name].hire.index(worker)
                self.agent_pool[name].hire.pop(fid)
                if config.Verbose: print('%s has been fired by %s'%(worker,name))
        if len(self.agent_pool[name].hire) == 0: # 解雇所有雇员,自己也失业但没有破产
            self.agent_pool[name].work = 0
            self.agent_pool[name].employer = None
        
        self.E, self.W, self.U = self.working_state() # 更新维护智能体状态

    def consume(self, name):
        # 消费，随机量m∈[0,m_a]
        # agent的钱减少m，市场价值增加m
        
        ma = self.agent_pool[name].coin
        if ma > 0: 
            m = np.random.randint(0, int(config.consume*ma))
        elif ma <= 0: 
            m = 0
            self.agent_pool[name].hungry += 1
        self.agent_pool[name].coin -= m
        self.market_V += m
        if config.die: 
            if self.agent_pool[name].hungry >= config.hungry_days:
                self.die(name) # 饥饿超过5步就死掉
        
    
    def die(self, name):
        '''死亡'''
        assert self.agent_pool[name].coin <= 0
        # self.agent_pool.pop(name)
        self.agent_pool[name].alive = False
        if name in self.E: idx = self.E.index(name); self.E.pop(idx)
        if name in self.U: idx = self.U.index(name); self.U.pop(idx)
        if name in self.W: idx = self.W.index(name); self.W.pop(idx)
        if config.Verbose: print('%s is dead!'%name)



env = Env()
env.reset()
for t in range(config.T):

    print(t, len(env.E),len(env.U),len(env.W),len(env.agent_pool),
        total_value(env.agent_pool,env.market_V),
        avg_coin(env.agent_pool,env.E),
        avg_coin(env.agent_pool,env.W))
    env.step(np.zeros((config.N)))


# env.add_agent()
