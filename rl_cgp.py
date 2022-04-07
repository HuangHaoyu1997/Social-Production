'''
利用RNN-based RL Policy生成符号解析式
'''
from itertools import count
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
def policy_generator(func_set=fs):
    '''
    return a sequence of symbols

    '''
    fs_dim = len(fs) # dimension of function set / categorical distribution
    tau = [] # sequence
    counter = 1

# test_lstm()

# print(cvt_bit(-1))
policy_generator()