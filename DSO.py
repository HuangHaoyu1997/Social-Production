from configuration import config
import numpy as np
import torch
from torch.distributions import Categorical
from utils import info_parser, pt_onehot, tanh, sigmoid

def policy_evaluator(tau, env, func_set, episode):
    '''
    policy evaluation
    policy is represented by a symbol sequence `tau`
    episode: test the policy for `config.Epoch` times, and average the episode reward
    '''
    
    global state
    r_epi = 0
    for i in range(episode):
        s = env.reset()
        state = s
        done = False
        reward = 0
        count = 0
        while not done:
            action = ComputingTree(tau, func_set, env_name)
            # for CartPoleContinuous
            # s, r, done, _ = env.step(np.array([action]))
            info, r, done = env.step(np.array([action]))
            s = info_parser(info)
            state = s
            reward += r
            count += 1
            if count >= config.max_step: break
        r_epi += reward
    return r_epi / episode

def ComputingTree(tau, func_set, env_name):
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
                if env_name == 'CartPole':
                    # [-1, 1] mapping for CartPoleContinuous
                    return tanh(out, alpha=0.1)
                elif env_name == 'SocialProduction':
                    # [0,50] mapping for Social-Production
                    return sigmoid(out, alpha=0.1)*50

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

def policy_generator(model, func_set,):
    '''
    return a sequence of symbols
    '''
    func_dim = len(func_set) # dimension of function set / categorical distribution
    tau = [] # symbol sequence

    # generte tau_1 with empty parent and sibling
    [iP, iS], P, S = ParentSibling(tau, func_set)
    PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
    
    counter = 1
    log_prob = 0
    joint_entropy = 0
    hn, cn = torch.zeros(2,1,config.hidden_size), torch.zeros(2,1,config.hidden_size)
    while counter > 0:
        phi, hn, cn = model(PS, hn, cn)
        
        mask = ApplyConstraints(tau, func_set)
        phi_after_mask = phi * mask
        phi_after_mask = phi_after_mask / phi_after_mask.sum()
        # print(phi,'\n',phi_after_mask,'\n\n')

        dist = Categorical(phi_after_mask[0,0])
        new_op = dist.sample()
        # print(new_op, phi_after_mask[0,0,new_op].log())
        log_prob += phi_after_mask[0,0,new_op].log()
        joint_entropy += dist.entropy()
        tau.append(new_op.item())
        
        PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
        counter += func_set[new_op].arity - 1
        if counter==0: break
        if len(tau) > config.N_COLS: return -1, 0, 0
        [iP, iS], P, S = ParentSibling(tau, func_set)
    
    if (func_dim-1 not in tau) and (func_dim-2 not in tau) and (func_dim-3 not in tau) and (func_dim-4 not in tau) and\
        (func_dim-5 not in tau) and (func_dim-6 not in tau) and (func_dim-7 not in tau) and (func_dim-8 not in tau) and \
        (func_dim-9 not in tau) and (func_dim-10 not in tau) and (func_dim-11 not in tau) and (func_dim-12 not in tau) and \
        (func_dim-13 not in tau) and (func_dim-14 not in tau) and (func_dim-15 not in tau) and (func_dim-16 not in tau) and \
        (func_dim-17 not in tau) and (func_dim-18 not in tau) and (func_dim-19 not in tau) and (func_dim-20 not in tau) and \
        (func_dim-21 not in tau) and (func_dim-22 not in tau) and (func_dim-23 not in tau) :
        return -1, 0, 0
    
    return tau, log_prob, joint_entropy

def ApplyConstraints(tau, func_set):
    '''
    给RNN输出的categorical概率施加约束
    如果parent是log/exp,则exp/log的概率为0
    如果parent是sin/cos,则cos/sin的概率为0
    '''
    # 如果tau空集合,不能选择常量作为根节点
    if len(tau)==0:
        mask = torch.tensor([0 if func_set[i].name in ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',\
                                                        's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',\
                                                        's21','s22'] else 1 for i in range(len(func_set))])
        return mask
    
    # 如果tau非空
    else:
        # iP是将要生成的node的parent在tau中的idx
        [iP,iS], P, S = ParentSibling(tau, func_set)
        # parent是iP在func_set中的idx
        parent = tau[iP]
        if func_set[parent].name == 'sin':
            mask = torch.tensor([0 if func_set[i].name=='cos' else 1 for i in range(len(func_set))])
        elif func_set[parent].name == 'cos':
            mask = torch.tensor([0 if func_set[i].name=='sin' else 1 for i in range(len(func_set))])
        elif func_set[parent].name == 'log':
            mask = torch.tensor([0 if func_set[i].name=='exp' else 1 for i in range(len(func_set))])
        elif func_set[parent].name == 'exp':
            mask = torch.tensor([0 if func_set[i].name=='log' else 1 for i in range(len(func_set))])
        else:
            mask = torch.ones(len(func_set))
        return mask