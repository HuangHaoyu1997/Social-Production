import numpy as np
from agent import agent
from configuration import config as c
import networkx as nx
import time, torch
import matplotlib.pyplot as plt
from numpy.random import uniform, randint
from scipy.spatial.distance import cdist

def coin_sort(agent_pool:dict, agent:list):
    '''
    计算指定agent群体的coin
    根据coin数量从低到高排序
    input: agent_pool是全体, agent是指定统计名单
    output: name_list, coin_list
    '''
    coin_list, name_list = [], []
    for name in agent:
        if agent_pool[name].alive:
            coin_list.append(agent_pool[name].coin)
            name_list.append(name)
    coin_list = np.array(coin_list)
    idx = coin_list.argsort()
    coin_list = coin_list[idx]
    name_list = [name_list[i] for i in idx]
    
    return name_list, coin_list

def avg_coin(agent_pool:dict, agent:list):
    '''
    计算指定agent群体的coin均值与方差
    '''
    if len(agent)==0: return 0., 0.
    _, _, mean, std, _ = financial_statistics(agent_pool, agent)
    return mean, std
    
def financial_statistics(agent_pool, agent):
    '''
    计算指定agent群体的coin最大数, 最小数, 平均数, 标准差, 中位数, 
    '''
    if alive_num(agent_pool, agent)==0: return 0., 0., 0., 0., 0.
    elif alive_num(agent_pool, agent)>0:
        _, coin_list = coin_sort(agent_pool, agent)
        return coin_list[-1], coin_list[0], coin_list.mean(), coin_list.std(), np.median(coin_list)

def grid_render(agent, resource):
    '''
    Grid可视化
    agent: env.agent_pool
    resource: 资源点的坐标
    '''
    x = c.x_range[1]
    y = c.y_range[1]
    max_coin, min_coin, avg_coin, _, _ = financial_statistics(agent, list(agent.keys())) # 最大财富值

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
            pixel_scale = min(coin / max_coin,0.2)
            grid[x,y,:] = int(255*pixel_scale)
    return grid

def info_parser(info):
    '''将env.reset()和env.step()输出的info字典解析为list,供CGP使用'''
    tmp = [info[name] for name in info]
    return tmp

def softmax(x):
    return np.exp(x)/(np.exp(x)).sum()

def tanh(x, alpha=1.0, with_clip=100):
    '''
    带有缩放因子alpha的tanh函数
    tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
    原函数的不饱和区间太窄,引入alpha<1对x进行缩放可以扩大不饱和区间
    '''
    x = np.clip(x, -with_clip, with_clip)
    return (np.exp(alpha*x)-np.exp(-alpha*x)) / (np.exp(alpha*x)+np.exp(-alpha*x))

def ComputingTree(tau, func_set):
    '''
    将操作符序列τ转化成一个计算树
    '''
    from cgp import Node
    assert np.max(tau) <= len(func_set) # 操作符必须全部来自func_set内
    
    # 按tau正向顺序创建操作符
    tree = []

    # 计算tau中各元素的parent
    parent = []
    l_tau = len(tau)
    for i in range(l_tau):
        [iP,iS], P, S = ParentSibling(tau[:i], func_set)
        parent.append(iP)
    assert (np.array(parent)==-1).sum() == 1 # 1个计算树只能有1个根节点,即只能有1个节点的parent=-1

    # 按照tau的正向顺序,创建计算节点
    for tau_i in tau:
        n = Node(func_set[tau_i].arity)
        n.i_func = tau_i
        tree.append(n)
    
    # 按照tau反向顺序,进行计算
    for i in reversed(range(len(tree))):
        if tree[i].arity==0:
            out = func_set[tree[i].i_func]()
            for j,k in enumerate(tree[parent[i]].i_inputs):
                if k is None:
                    tree[parent[i]].i_inputs[j] = out
                    break
        elif tree[i].arity>0:
            assert (np.array(tree[i].i_inputs)==None).sum()==0 # i的input buf已经存满了i节点的子节点传给i的结果
            out = func_set[tree[i].i_func](*tree[i].i_inputs)
            # 如果是根节点,直接return
            if parent[i] == -1: 
                # [-1, 1] mapping
                return tanh(out, alpha=0.1)
            for j,k in enumerate(tree[parent[i]].i_inputs):
                if k is None:
                    tree[parent[i]].i_inputs[j] = out
                    break

def ParentSibling(tau, func_set):
    '''
    tau: 输入的符号序列,tau中每个元素是function_set中的序号,序号从0开始计数
    function_set: 符号库
    return: 
    输出下一个算符的parent和sibling在tau中的idx和在func_set中的idx的onehot向量。
    注意, 这个元素还不在tau里, 还没被生成出来
    parent或sibling为空, 返回全0向量
    
    '''
    T = len(tau)
    counter = 0
    template = torch.zeros(len(func_set))
    
    if T == 0:
        return [-1, -1], template, template
    
    if func_set[tau[T-1]].arity > 0:
        # print(func_set[tau[T-1]].name)
        parent = tau[T-1]
        sibling = -1
        
        return [T-1, -1], pt_onehot(x=[parent], dim=len(func_set))[0], template

    for i in reversed(range(T)):
        counter += (func_set[tau[i]].arity - 1)
        if counter == 0:
            parent = tau[i]
            sibling = tau[i+1]
            return [i, i+1], pt_onehot(x=[parent], dim=len(func_set))[0], pt_onehot(x=[sibling], dim=len(func_set))[0]

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
    
    # tmp = cdist([[agent.x, agent.y]], resource)
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
    idx = idx[0:N]
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
    return total_ex / total_la if total_la>0 else 0 # total_ratio / count, 

def build_graph(agent):
    '''
    建立关系网络, 用于reset()函数
    '''
    G = nx.Graph()
    for name in agent:
        if agent[name].alive:
            G.add_node(name,coin=agent[name].coin)
    return G

def add_agent(N, flag=None):
    '''
    添加智能体
    flag:指示是否初次创建人口
    None表示初次创建,人口年龄服从高斯分布
    flag=1表示非初次,人口年龄为18-20岁
    '''
    pool = {}
    for i in range(N):
        # 用CPU时钟的小数位命名智能体
        name = str(i)+str(time.time()).split('.')[1] # 【应该不会出现重名？】
        
        # 随机坐标
        x = round(uniform(c.x_range[0],c.x_range[1]),2)
        y = round(uniform(c.y_range[0],c.y_range[1]),2)
        
        # [0,1]均匀分布
        # skill = round(uniform(c.skill[0], c.skill[1]),2)
        # [0,1]区间,均值0.5,标准差0.2的截断高斯分布
        # mean = 0.5*(c.skill[0] + c.skill[1])
        # skill = round(np.clip(np.random.randn()*0.2+mean, c.skill[0], c.skill[1]),3)
        
        # [0,1]之间,均值0.5的Beta分布,mean=a/(a+b),a=b越小分布越均匀
        skill = round(np.random.beta(a=5,b=5))

        coin = uniform(c.coin_range[0],c.coin_range[1]) if c.random_coin else c.init_coin
        
        # 初始年龄分布是[15,75]之间,均值38,标准差10的截断高斯分布,中美两国平均年龄均38岁
        if flag is None:
            age = np.clip((np.random.randn()*10+c.age_mean), 15, 100)
        if flag == 1:
            age = round(uniform(18,20),2)
        intention = c.employment_intention if age>=18 and age<c.retire_age else 0
        pool[name] = agent(x, y, name, skill, coin, age, intention)
    return pool

def avg_age(agent_pool):
    '''
    统计活人的平均年龄
    '''
    count = []
    for name in agent_pool:
        if agent_pool[name].alive:
            count.append(agent_pool[name].age)
    return np.mean(count)

def alive_num(agent_pool, agent=None):
    if agent is None:
        count = 0
        for name in agent_pool:
            if agent_pool[name].alive:
                count += 1
    elif agent is not None:
        count = 0
        for name in agent:
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
def discounted_weight(length):
    '''
    生成长度=length的折扣权重
    用于计算历史工资发放记录的加权平均
    离现在越近的工资,weight越大,越早的工资发放记录,weight越小
    '''
    assert length <= c.salary_deque_maxlen
    if length==1: return np.array(1.0)
    salary_weight = [pow(c.salary_gamma, length-i) for i in range(length)]
    salary_weight = np.array(salary_weight) / np.array(salary_weight).sum() # normalized
    # salary_weight = np.exp(salary_weight) / np.exp(salary_weight).sum() # softmax
    return salary_weight

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
                    assert agent[worker].work == 2
                    worker_num += 1
                    # 若该worker及其雇佣者之间没有edge
                    if not G.has_edge(name,worker):
                        G.add_edge(name,worker)
            
            # 失业者的edge必须全部删去
            elif agent[name].work == 0:
                for n_adj in list(G[name]): # G[name]得到name的邻居
                    G.remove_edge(n_adj,name)
            
            # TODO 
            # 【想法】根据agent的资本量建立edge，将其[-20%,20%]范围内的智能体建立连接
            
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
        
        测试代码
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
        

def pt_onehot(x, dim):
    '''
    生成pytorch tensor onehot向量
    dim: onehot向量的维度

    示例代码:
    pt_onehot([1],8)
    '''
    

    index = torch.tensor(x)
    length = len(index)
    a = index.unsqueeze(1)
    result = torch.zeros(length, dim).scatter_(1,a,1)
    return result 


'''
input_dim = 10
out_dim = 5
pop = create_population(5,input_dim,out_dim)

def cvt_bit(a,length=6):
    # 将int a转化为定长二进制str
    return '0'*(length-len(bin(a)[2:]))+bin(a)[2:]

def gene_encoder(ind:Individual):
    # 打印个体的节点
    
    gene_before = []
    gene_after = ''
    for node in ind.nodes:
        # print(node.i_func, cvt_bit(node.i_func), int(cvt_bit(node.i_func),2))
        # print(node.i_inputs[0]+10, cvt_bit(node.i_inputs[0]+10), int(cvt_bit(node.i_inputs[0]+10),2))
        gene_after += cvt_bit(node.i_func)
        gene_after += cvt_bit(node.i_inputs[0]+input_dim)
        
        gene_before.append(node.i_func)
        gene_before.append(node.i_inputs[0]+input_dim)
        
        if node.i_inputs[1] is not None:
            # print(node.i_inputs[1]+10, cvt_bit(node.i_inputs[1]+10))
            gene_after += cvt_bit(node.i_inputs[1]+input_dim)
            gene_before.append(node.i_inputs[1]+input_dim)
    return gene_before, gene_after

g_b, g_a = gene_encoder(pop[0])
'''



if __name__ == "__main__":
    pass