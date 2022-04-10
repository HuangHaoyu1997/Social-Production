'''
利用RNN-based RL Policy生成符号解析式
'''
from urllib.parse import ParseResult
import torch
import torch.nn as nn
from cgp import *

def test_lstm():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    func_dim = 10 # 操作符字典的长度
    batch_size = 1
    num_layer = 5
    hidden_dim = 16

    # input vec len(x)=10, lstm hidden layer dim=16, 此lstm model用2个lstm层。如果是1，可以省略，默认为1)
    rnn = nn.LSTM(func_dim, hidden_dim, num_layer).to(device) 
    fc = nn.Linear(hidden_dim, func_dim)
    # 初始化hidden vec和memory vec, 通常其维度相同
    # 2个LSTM层，batch_size=3, 隐藏层的特征维度20
    h0 = torch.randn(num_layer, batch_size, hidden_dim).to(device)
    c0 = torch.randn(num_layer, batch_size, hidden_dim).to(device)

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
    fs_dim = len(func_set) # dimension of function set / categorical distribution
    tau = [] # sequence
    counter = 1
    
    # generte tau_1 with empty parent and sibling
    parent = torch.zeros(1, batch_size, fs_dim)
    sibling = torch.zeros(1, batch_size, fs_dim)
    h0 = torch.randn(num_layer, batch_size, hidden_dim).to(device)
    c0 = torch.randn(num_layer, batch_size, hidden_dim).to(device)
    print(torch.cat((parent,sibling),dim=-1).shape)

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
    
]
# ÷ sin × c01 const_1 log 
# 3, 4, 2, 8, 9
tau = [3,4,2,8,9,] # 6
def ParentSibling(tau, function_set):
    '''
    tau: 输入的符号序列
    function_set: 符号库
    return: 输出下一个元素的parent和sibling，注意，这个元素还不在tau里，还没被生成出来
    '''
    T = len(tau)
    counter = 0
    if T == 0:
        return -1, -1
    if function_set[tau[T-1]].arity > 0:
        # print(function_set[tau[T-1]].name)
        parent = tau[T-1]
        sibling = -1
        return parent, sibling
    for i in reversed(range(T)):
        counter += (function_set[tau[i]].arity - 1)
        if counter == 0:
            parent = tau[i]
            sibling = tau[i+1]
            return parent, sibling


# test_lstm()

# print(cvt_bit(-1))
# policy_generator()
print(ParentSibling([],fs))
print(ParentSibling([3],fs))
print(ParentSibling([3,4],fs))
print(ParentSibling([3,4,2],fs))
print(ParentSibling([3,4,2,8],fs))
print(ParentSibling([3,4,2,8,9],fs))
print(ParentSibling([3,4,2,8,9,6],fs))