import numpy as np
from agent import agent
from configuration import config

def avg_coin(agent_pool:dict, agent:list):
    '''
    计算指定agent群体的财富均值与方差
    '''
    if len(agent)==0: return 0., 0.
    count = []
    for name in agent:
        if agent_pool[name].alive:
            count.append(agent_pool[name].coin)
    count = np.array(count)
    return np.round(count.mean(),2), np.round(count.std(),2)

def total_value(agent_pool, V):
    '''
    return: agent总货币量, 市场价值, 系统总价值
    '''
    M = 0
    for name in agent_pool:
        if agent_pool[name].alive:
            M += agent_pool[name].coin
    return M,V,M+V

def CalDistance(agent:agent,resource:np.ndarray):
    '''
    返回指定智能体与所有资源点之间的最小距离
    '''
    tmp = np.sqrt((agent.x - resource[:,0])**2 + (agent.y - resource[:,1])**2)
    return tmp.min()

def render():
