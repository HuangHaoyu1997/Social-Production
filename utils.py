from ast import Return
import numpy as np
from pandas import array
from agent import agent
from configuration import config
import networkx as nx
import cgp

def max_coin(agent):
    max_income = 0
    for name in agent:
        if agent[name].alive:
            if agent[name].coin >= max_income:
                max_income = agent[name].coin
    return max_income

def grid_render(agent,resource):
    x = config.x_range[1]
    y = config.y_range[1]
    max_income = max_coin(agent)

    grid = np.zeros((x+1,y+1,3),dtype=np.uint8)
    for rx,ry in resource:
        rx = round(rx)
        ry = round(ry)
        grid[rx-5:rx+5,ry-5:ry+5,:] = [0,255,0]

    for name in agent:
        if agent[name].alive:
            x = round(agent[name].x)
            y = round(agent[name].y)
            coin = agent[name].coin
            pixel_scale = min(coin / max_income,0.2)
            grid[x,y,:] = int(255*pixel_scale)
    return grid

def extract_computational_subgraph(ind: cgp.Individual) -> nx.MultiDiGraph:
    """Extract a computational subgraph of the CGP graph `ind`, which only contains active nodes.

    Args:
        ind (cgp.Individual): an individual in CGP  

    Returns:
        nx.DiGraph: a acyclic directed graph denoting a computational graph

    See https://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/ and 
    http://www.cs.columbia.edu/~mcollins/ff2.pdf for knowledge of computational graphs.
    """
    # make sure that active nodes have been confirmed
    if not ind._active_determined:
        ind._determine_active_nodes()
        ind._active_determined = True
    # in the digraph, each node is identified by its index in `ind.nodes`
    # if node i depends on node j, then there is an edge j->i
    g = nx.MultiDiGraph()  # possibly duplicated edges
    for i, node in enumerate(ind.nodes):
        if node.active:
            f = ind.function_set[node.i_func]
            g.add_node(i, func=f.name)
            order = 1
            for j in range(f.arity):
                i_input = node.i_inputs[j]
                w = node.weights[j]
                if i_input == -3: i_input = 'consume'
                if i_input == -2: i_input = 'coin_before'
                if i_input == -1: i_input = 'hungry_before'
                g.add_edge(i_input, i, weight=w, order=order)
                order += 1
    return g

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
    pool = {}
    for i in range(N):
        name = str(i)
        pool[name] = agent(name)
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
