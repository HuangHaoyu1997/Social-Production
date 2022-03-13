import numpy as np

def avg_coin(agent_pool:dict, agent:list):
    '''
    计算指定agent群体的财富均值与方差
    '''
    if len(agent)==0: return 0., 0.
    count = []
    for name in agent:
        count.append(agent_pool[name].coin)
    count = np.array(count)
    return np.round(count.mean(),2), np.round(count.std(),2)

def total_value(agent_pool, V):
    '''
    return: agent总货币量, 市场价值, 系统总价值
    '''
    M = 0
    for name in agent_pool:
        M += agent_pool[name].coin
    return M,V,M+V