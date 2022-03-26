import torch
import torch.nn as nn
from cgp import *

def test_lstm():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    rnn = nn.LSTM(10, 20, 2).to(device) 
    # 输入数据x的向量维数10, 设定lstm隐藏层的特征维度20, 此model用2个lstm层。如果是1，可以省略，默认为1)
    input = torch.randn(5, 3, 10).to(device)
    # 输入的input为，序列长度seq_len=5, 每次取的minibatch大小，batch_size=3, 数据向量维数=10（仍然为x的维度）。每次运行时取3个含有5个字的句子（且句子中每个字的维度为10进行运行）
    # 初始化的隐藏元和记忆元,通常它们的维度是一样的
    # 2个LSTM层，batch_size=3, 隐藏层的特征维度20
    h0 = torch.randn(2, 3, 20).to(device)
    c0 = torch.randn(2, 3, 20).to(device)
    # 这里有2层lstm，output是最后一层lstm的每个词向量对应隐藏层的输出,其与层数无关，只与序列长度相关
    # hn,cn是所有层最后一个隐藏元和记忆元的输出
    output, (hn, cn) = rnn(input, (h0, c0))
    ##模型的三个输入与三个输出。三个输入与输出的理解见上三输入，三输出
    #输出：torch.Size([5, 3, 20]) torch.Size([2, 3, 20]) torch.Size([2, 3, 20])

    # print(output.size(),hn.size(),cn.size(),output.device)

pop = create_population(5,input_dim=10,out_dim=5)

def print_ind(ind:Individual):
    '''
    打印个体的节点
    '''
    for node in ind.nodes:
        # i_func_bin = bin(node.i_func).replace('0b','')
        i_func_bin = '0'*(5-len(bin(node.i_func)[2:]))+bin(node.i_func)[2:]
        print(type(i_func_bin))

print_ind(pop[0])