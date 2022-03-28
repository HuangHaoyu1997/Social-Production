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

input_dim = 10
out_dim = 5
pop = create_population(5,input_dim,out_dim)


def cvt_bit(a,length=6):
    '''
    将int a转化为定长二进制str
    '''
    return '0'*(length-len(bin(a)[2:]))+bin(a)[2:]

def gene_encoder(ind:Individual):
    '''
    打印个体的节点
    '''
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
        '''
        
        '''
    return gene_before, gene_after

g_b, g_a = gene_encoder(pop[0])

# print(cvt_bit(-1))