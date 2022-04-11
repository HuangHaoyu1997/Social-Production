'''
利用RNN-based RL Policy生成符号解析式
'''

import torch
import torch.nn as nn
from cgp import *
from utils import pt_onehot
from torch.distributions import Categorical

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
    Function(const_1, 0),       # 9
    Function(const_5, 0),       # 10
]

class lstm(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layer):
        super(lstm,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hn=None, cn=None):
        if hn is not None and cn is not None:
            x, (hn, cn) = self.lstm(x, (hn, cn))
        else:
            x, (hn, cn) = self.lstm(x)
        s, b, h = x.size() # s序列长度,b批大小,h隐层维度
        # print(s,b,h,hn.shape,cn.shape)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        x = torch.softmax(x, dim=-1)
        return x, hn, cn

def test_lstm():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    func_dim = 10 # 操作符字典的长度
    batch_size = 1
    num_layer = 5
    hidden_dim = 16

    # input vec len(x)=10, lstm hidden layer dim=16, 此lstm model用2个lstm层。如果是1，可以省略，默认为1)
    model = lstm(input_size = 2*func_dim,
                    hidden_size = hidden_dim, 
                    output_size = func_dim, 
                    num_layer = num_layer
                ).to(device)
    # 初始化hidden vec和memory vec, 通常其维度相同
    # 2个LSTM层，batch_size=3, 隐藏层的特征维度20
    h0 = torch.zeros(num_layer, batch_size, hidden_dim).to(device)
    c0 = torch.zeros(num_layer, batch_size, hidden_dim).to(device)

    # input seq_len=5, batch_size=3, len(x)=10, 每次运行时取3个含有5个word的seq,每个word的维度为10
    input = torch.randn(1, batch_size, func_dim).to(device)
    for i in range(5):
        
        # 这里有2层lstm，output是最后一层lstm的每个词向量对应隐藏层的输出,其与层数无关，只与序列长度相关
        # hn,cn是各层最后一步的hidden vec和memory vec的输出
        # torch.Size([5, 1, 20]) torch.Size([2, 1, 20]) torch.Size([2, 1, 20])
        output, (hn, cn) = rnn(input, (h0, c0))
        print(output[-1].unsqueeze(0).shape)
        output = fc(output)
        
        print(input.shape,output.size(),hn.size(),cn.size(),output.device)
        input = torch.cat((input, output))

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
def policy_generator(num_layer=2, hidden_dim=32, batch_size=1, func_set=fs, device=torch.device('cpu')):
    '''
    return a sequence of symbols

    '''
    func_dim = len(func_set) # dimension of function set / categorical distribution
    tau = [] # symbol sequence
    # generte tau_1 with empty parent and sibling
    [iP, iS], P, S = ParentSibling(tau, func_set)
    PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
    counter = 1
    hn, cn = None, None
    # lstm model
    model = lstm(input_size = 2*func_dim,
                    hidden_size = hidden_dim, 
                    output_size = func_dim, 
                    num_layer = num_layer
                ).to(device)
    while counter > 0:
        phi, hn, cn = model(PS, hn, cn)
        
        mask = ApplyConstraints(tau, func_set)
        phi *= mask
        phi = phi / phi.sum()

        dist = Categorical(phi[0,0])
        new_op = dist.sample()
        tau.append(new_op.item())
        
        
        
        
        PS = torch.cat((P,S)).unsqueeze(0).unsqueeze(0)
        counter += func_set[new_op].arity - 1
        if counter==0: break
        [iP, iS], P, S = ParentSibling(tau, func_set)
        print(tau)
    return tau



# ÷ sin × c01 const_1 log 
# 3, 4, 2, 8, 9
tau = [3,4,2,8,9,] # 6

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

def ApplyConstraints(tau, func_set):
    '''
    给RNN输出的categorical概率施加约束
    如果parent是log/exp,则exp/log的概率为0
    如果parent是sin/cos,则cos/sin的概率为0
    '''
    # 如果tau空集合,不能选择常量作为根节点
    if len(tau)==0:
        
        mask = torch.tensor([0 if func_set[i].name in ['const_01','const_1','const_5'] else 1 for i in range(len(func_set))])
        return mask
    
    # 如果tau非空
    else:
        # 计算parent
        [iP,iS], P, S = ParentSibling(tau, func_set)
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
    
def ComputingTree(tau, func_set):
    '''
    将操作符序列τ转化成一个计算树
    '''
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
            if parent[i] == -1: return out
            for j,k in enumerate(tree[parent[i]].i_inputs):
                if k is None:
                    tree[parent[i]].i_inputs[j] = out
                    break


# test_lstm()

# print(cvt_bit(-1))
tau = policy_generator()

print(len(tau), tau)
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
