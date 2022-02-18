from tkinter import W
import numpy as np
from config import config
from agent import agent
import random

class env:
    def __init__(self,param) -> None:
        self.param = param
        
    
    def reset(self,):
        self.t = 0
        self.agent_num = self.param.N
        self.agent_pool = self.add_agent(self.agent_num)
        self.market_V = self.param.V
        self.E, self.W, self.U = self.working_state()

        return [self.E, self.W ,self.U, self.market_V]

    def add_agent(self, N):
        pool = {}
        for i in range(N):
            param = config()
            param.name = str(i)
            pool[param.name] = agent(param)
        return pool
    
    def working_state(self,):
        # 统计当前所有agent工作状态
        E, W, U = [], [], []
        for name in self.agent_pool:
            if self.agent_pool[name].work == 0:
                U.append(name)
            elif isinstance(self.agent_pool[name].work, str):
                if self.agent_pool[name].work == name:
                    # 雇主是自己
                    E.append(name)
                elif self.agent_pool[name].work != name:
                    # 雇主不是自己
                    W.append(name)
        return E, W, U
    
    def hire(self,):
        pass

    def step(self,):
        # 单步执行函数
        
        pass
    
    def pay(self,):
        # 遍历E中的每个agent a，为a.hire中的工人发工资
        
        for employer in self.E:
            capital = self.agent_pool[employer].coin
            worker_list = self.agent_pool[employer].hire
            random.shuffle(worker_list)
            for worker in worker_list:
                w = np.random.randint(self.param.w1,self.param.w2)
                if w <= capital:
                    self.agent_pool[worker].coin += w
                    capital -= w
                if w > capital:
                    
        return None

    def broken(self):
        pass
        
        return None


def total_value(agent_pool, V):
    # 系统总货币量
    M = 0
    for name in agent_pool:
        M += agent_pool[name].coin
    return M+V

def consume(agent_pool):
    # 消费，随机量m∈[0,m_a]
    # agent的钱减少m，市场价值增加m
    V_value = 0
    for name in agent_pool:
        ma = agent_pool[name].coin
        m = np.random.randint(0,ma)
        agent_pool[name].coin -= m
        V_value += m
    return agent_pool, V_value

def salary(agent):
    pass



env = env(config)
# env.add_agent()
