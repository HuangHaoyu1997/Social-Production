import numpy as np
from agent import agent
from configuration import config as c
import networkx as nx
import cgp, time
import matplotlib.pyplot as plt
from numpy.random import uniform

def max_coin(agent):
    '''
    计算指定agent群体中的最大收入
    '''
    max_income = 0
    for name in agent:
        if agent[name].alive:
            if agent[name].coin >= max_income:
                max_income = agent[name].coin
    return max_income

def grid_render(agent,resource):
    x = c.x_range[1]
    y = c.y_range[1]
    max_income = max_coin(agent)

    grid = np.zeros((x+1,y+1,3),dtype=np.uint8)
    for rx,ry in resource:
        rx = int(round(rx))
        ry = int(round(ry))
        grid[rx-2:rx+2,ry-2:ry+2,:] = [0,255,0]

    for name in agent:
        if agent[name].alive:
            x = int(round(agent[name].x))
            y = int(round(agent[name].y))
            coin = agent[name].coin
            # 灰度值与收入正相关
            pixel_scale = min(coin / max_income,0.2)
            grid[x,y,:] = int(255*pixel_scale)
    return grid


def softmax(x):
    return np.exp(x)/(np.exp(x)).sum()

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

def total_value(agent_pool, V, G):
    '''
    return: agent总货币量, 市场价值, 系统总价值
    '''
    M = 0
    for name in agent_pool:
        if agent_pool[name].alive:
            M += agent_pool[name].coin
    return M,V,G,M+V+G

def CalDistance(agent:agent,resource:np.ndarray):
    '''
    返回指定智能体与所有资源点之间的最小距离
    '''
    tmp = np.sqrt((agent.x - resource[:,0])**2 + (agent.y - resource[:,1])**2)
    return tmp.min()

def most_poor(agent, N):
    '''
    找出最穷的N个人
    out: 最穷的N个人的name,按最穷-->次穷的顺序
    '''
    poor_list = []
    name_list = []
    out = []
    for name in agent:
        if agent[name].alive:
            poor_list.append(agent[name].coin)
            name_list.append(name)
    poor_list = np.array(poor_list)
    idx = poor_list.argsort()[:N]
    for i in idx:
        out.append(name_list[i])
    return out

def most_rich(agent, N):
    '''
    找出最富的N个人
    out: 最富的N个人的name,按最富-->次富的顺序
    '''
    rich_list = []
    name_list = []
    out = []
    for name in agent:
        if agent[name].alive:
            rich_list.append(agent[name].coin)
            name_list.append(name)
    rich_list = np.array(rich_list)
    idx = rich_list.argsort()[::-1]
    for i in idx:
        out.append(name_list[i])
    return out

def exploit_ratio(agent, employer):
    '''
    计算平均剩余价值率=剩余价值/可变资本（劳力成本）
    '''
    total_ex = 0
    total_la = 0
    total_ratio = 0
    count = 0
    if len(employer) == 0: return 0.
    for name in employer:
        assert agent[name].work == 1
        assert agent[name].alive is True
        # if agent[name].labor_cost ==0:
        # ratio = agent[name].exploit / agent[name].labor_cost
        total_ex += agent[name].exploit
        total_la += agent[name].labor_cost
        # total_ratio += ratio
        # count += 1
    return total_ex / total_la # total_ratio / count, 

def build_graph(agent):
    G = nx.Graph()
    for name in agent:
        if agent[name].alive:
            G.add_node(name,coin=agent[name].coin)
    return G

def add_agent(N):
    '''
    添加智能体
    '''
    pool = {}
    for i in range(N):
        # 用CPU时钟的小数位命名智能体
        name = str(i)+str(time.time()).split('.')[1]
        x = round(uniform(c.x_range[0],c.x_range[1]),2)
        y = round(uniform(c.y_range[0],c.y_range[1]),2)
        
        if c.skill_gaussian:
            mean = 0.5*(c.skill[0] + c.skill[1])
            skill = round(np.clip(np.random.randn()+mean, c.skill[0], c.skill[1]),3)
        else:
            skill = round(uniform(c.skill[0], c.skill[1]),2)
        coin = uniform(c.coin_range[0],c.coin_range[1]) if c.random_coin else c.init_coin
        pool[name] = agent(x, y, name, skill, coin)
    return pool

def alive_num(agent_pool):
    count = 0
    for name in agent_pool:
        if agent_pool[name].alive:
            count += 1
    return count

def working_state(agent_pool):
    '''
    统计当前所有agent工作状态
    E:Employer
    W:Worker
    U:Unemployed
    '''

    E, W, U = [], [], []
    for name in agent_pool:
        if agent_pool[name].alive:
            if agent_pool[name].work == 0:# 失业者
                U.append(name)
            elif agent_pool[name].work == 1:# 雇主是自己，资本家
                E.append(name)
            elif agent_pool[name].work == 2:# 雇主是别人，自己是工人
                W.append(name)
    return E, W, U

def update_graph(G:nx.Graph, agent:dict, E, W, U):
    '''
    根据雇佣关系更新图网络
    '''
    worker_num, employer_num = 0, 0
    for name in agent:
        # TODO
        # 后续工作应该将死掉的节点改变颜色，如灰色，继续存在于图中


        # 将死人从Graph中删除
        if not agent[name].alive:
            if G.has_node(name):
                G.remove_node(name)

        if agent[name].alive:
            # 检查所有活人都在Graph中
            if not G.has_node(name):
                G.add_node(name, coin=agent[name].coin)
            
            # 以资本家为线索，建立雇佣者与被雇者之间的Graph，失业者作为孤立点存在
            if agent[name].work == 1:
                employer_num += 1
                for worker in agent[name].hire:
                    assert agent[worker].alive is True
                    worker_num += 1
                    # 若该worker及其雇佣者之间没有edge
                    if not G.has_edge(name,worker):
                        G.add_edge(name,worker)
            
            # 失业者的edge必须全部删去
            elif agent[name].work == 0:
                for n_adj in list(G[name]): # G[name]得到name的邻居
                    G.remove_edge(n_adj,name)
    assert worker_num==len(W) and employer_num==len(E)
    return G



class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=1.0, theta=0.1, dt=0.1, x0=5.):
        '''
        OU过程是均值回归过程,当x_t比均值μ大时,下一步状态值x_{t+△t}就会变小;反之,x_{t+△t}会变大。
        简单地说就是状态值x_t偏离均值μ时会被拉回。
        OU过程具有时序相关性,而高斯随机过程的相邻两个时间步前后无关。
        OU noise往往不会高斯噪声一样相邻的两步的值差别那么大,而是会绕着均值附近正向或负向探索一段距离，
        就像物价和利率的波动一样，这有利于在一个方向上探索。

        mu: asymptotic mean,渐进均值
        sigma: 噪声强度, 越大波动越大, 曲线毛刺越多, OU过程的噪声是一个维纳过程(布朗运动), 每一时间间隔内的噪声服从高斯分布
        theta: how strongly the system reacts to perturbations(the decay-rate or growth-rate), 向均值回归的速度, theta越小,波动越大
        dt: 时间分辨率/时间尺度, 值越小, 变化越慢。
        x0: 状态初始值
        '''
        self.theta = theta
        self.mu = np.array(mu) # np.array允许mu是实数,而不仅len>=2的向量
        self.sigma = sigma
        self.dt = dt
        self.x0 = np.array(x0)
        self.reset()

    def __call__(self):
        x = self.x_previous + self.theta * (self.mu - self.x_previous) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_previous = x
        return x

    def reset(self):
        self.x_previous = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)







if __name__ == "__main__":
    ou_noise1 = OrnsteinUhlenbeckActionNoise(mu=c.w1,theta=0.1,sigma=5,x0=10,dt=0.1)
    ou_noise2 = OrnsteinUhlenbeckActionNoise(mu=c.w1,theta=0.1,sigma=5,x0=10,dt=0.1)
    # plt.figure()
    y1 = []
    y2 = [] # np.random.normal(0, 1, 10000)
    # t = np.linspace(0, 1000, 10000)
    
    for _ in range(1200):
        y1.append(ou_noise1())
        y2.append(ou_noise2())
        # if y1[-1]<=0: print(y1[-1])
        # y2.append(ou_noise2())

    plt.plot(y1, c='r')
    plt.plot(y2, c='b')
    plt.show()