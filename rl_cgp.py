'''
利用RNN-based RL Policy生成符号解析式
'''

import torch
import torch.nn as nn
from cgp import *
from utils import ParentSibling, ComputingTree
from torch.distributions import Categorical
from CartPoleContinuous import CartPoleContinuousEnv

device = torch.device('cpu')


env = CartPoleContinuousEnv()
state = env.reset()

def s0():
    '''返回env状态的第0维度'''
    return state[0]
def s1():
    '''返回env状态的第1维度'''
    return state[1]
def s2():
    '''返回env状态的第2维度'''
    return state[2]
def s3():
    '''返回env状态的第3维度'''
    return state[3]
fs = [
    Function(op.add, 2),        # 0
    Function(op.sub, 2),        # 1
    Function(op.mul, 2),        # 2
    Function(protected_div, 2), # 3
    Function(math.sin, 1),      # 4
    Function(math.cos, 1),      # 5
    Function(math.log, 1),      # 6
    Function(math.exp, 1),      # 7
    Function(const_01, 0),      # 8
    # Function(const_1, 0),       # 9
    # Function(const_5, 0),       # 10
    Function(s0, 0),
    Function(s1, 0),
    Function(s2, 0),
    Function(s3, 0),
]

def test_gym():
    env.reset()
    done = False
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        global state
        state = s
        print(fs[-4](),fs[-3](),fs[-2](),fs[-1](),'\n')
# test_gym()


class lstm(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layer):
        super(lstm,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hn=None, cn=None):
        if hn is not None and cn is not None:
            x, (hn, cn) = self.lstm(x, (hn, cn))
        else: x, (hn, cn) = self.lstm(x)
        s, b, h = x.size() # s序列长度, b批大小, h隐层维度
        # print(s,b,h,hn.shape,cn.shape)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        x = torch.softmax(x, dim=-1)
        return x, hn, cn

def policy_evaluator(tau, env, func_set=fs, episode=config.Epoch):
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
        while not done:
            action = ComputingTree(tau, func_set)
            s, r, done, _ = env.step(np.array([action]))
            state = s
            reward += r
        r_epi += reward
    return r_epi / episode


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

# lstm model
model = lstm(input_size = 2*len(fs),
                hidden_size = 32, 
                output_size = len(fs), 
                num_layer = 2
            ).to(device)

def policy_generator(model, func_set=fs,):
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
    hn, cn = torch.zeros(2,1,32), torch.zeros(2,1,32)
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
        tau.append(new_op.item())
        
        PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
        counter += func_set[new_op].arity - 1
        if counter==0: break
        if len(tau) > config.N_COLS: return -1, 0
        [iP, iS], P, S = ParentSibling(tau, func_set)
    if (func_dim-1 not in tau) and (func_dim-2 not in tau) and (func_dim-3 not in tau) and (func_dim-4 not in tau):
        return -1, 0
    return tau, log_prob

def ApplyConstraints(tau, func_set):
    '''
    给RNN输出的categorical概率施加约束
    如果parent是log/exp,则exp/log的概率为0
    如果parent是sin/cos,则cos/sin的概率为0
    '''
    # 如果tau空集合,不能选择常量作为根节点
    if len(tau)==0:
        mask = torch.tensor([0 if func_set[i].name in ['s0','s1','s2','s3'] else 1 for i in range(len(func_set))])
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
    '''
    elif len(tau)>0:
        par
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

    '''
    


'''tau_len = []
for i in range(1):
    tau = policy_generator(model)
    if tau != -1:
        tau_len.append(len(tau))

print(tau_len, len(tau_len), np.mean(tau_len))'''
tau = -1
while tau==-1:
    tau, log_prob = policy_generator(model)
R = policy_evaluator(tau, env, fs)
print(tau, R)

'''
print(ParentSibling([],fs))
print(ParentSibling([3],fs))
print(ParentSibling([3,4],fs))
print(ParentSibling([3,4,2],fs))
print(ParentSibling([3,4,2,8],fs))
print(ParentSibling([3,4,2,8,9],fs))


'''
# print(ParentSibling([3,4,2,8,9,6], fs))
'''model =lstm(10,8,5,2)
x, hn, cn = model(torch.rand(1,1,10), ) # torch.zeros(3,1,8), torch.zeros(3,1,8)
print(x.shape, hn, cn)'''

'''
import time
print(ComputingTree([3,4,2,8,9,6,10], fs))
tick = time.time()
for i in range(100):
    pass; #ComputingTree([0,1,3,8,10,4,9,2,5,7,8,6,10], fs)
print((time.time()-tick)/100*1000,'ms')
'''