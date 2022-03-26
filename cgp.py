"""
Cartesian genetic programming
"""
from ast import Lambda

import random
import copy
import math
from sre_parse import Verbose
import numpy as np
from function import *
from configuration import config

np.random.seed(config.seed)
random.seed(config.seed)

class Node:
    """
    A node in CGP graph
    """
    def __init__(self, max_arity):
        """
        Initialize this node randomly
        """
        
        self.i_func = None # 该节点的函数在函数集的index
        self.i_inputs = [None] * max_arity
        self.weights = [None] * max_arity
        self.i_output = None
        self.output = None
        self.active = False


class Individual:
    """
    An individual (chromosome, genotype, etc.) in evolution
    
    """
    function_set = None
    weight_range = [-1, 1]
    max_arity = 3
    
    n_cols = config.N_COLS # number of cols (nodes) in a single-row CGP
    level_back = config.LEVEL_BACK # 后面的节点可以最远连接的前面节点的相对位置
    fitness = None

    def __init__(self,input_dim,out_dim):
        self.n_inputs = input_dim
        self.n_outputs = out_dim # 输出维度
        self.nodes = []
        for pos in range(self.n_cols):
            self.nodes.append(self._create_random_node(pos))
        
        # 将最后n_outputs个node设为输出节点
        for i in range(1, self.n_outputs + 1):
            self.nodes[-i].active = True
        
        self.fitness = None
        self._active_determined = False

    def _create_random_node(self, pos):
        '''
        pos:该节点的index
        设: n_inputs=3,level_back=4

        in  in  in  0   1   2   3   4   5   6
        *   *   *   *   *   *   *   *   *   *

        pos  pos-level_back  -n_inputs  max(p-l,-n)  pos-1  i_inputs取值
        0        -4              -3         -3         -1    -3,-2,-1
        1        -3              -3         -3          0    -3,-2,-1,0
        2        -2              -3         -2          1    -2,-1,0,1
        3        -1              -3         -1          2    -1,0,1,2
        4         0              -3          0          3     0,1,2,3
        5         1              -3          1          4     1,2,3,4
        6         2              -3          2          5     2,3,4,5
        
        输入维度=3,则-3,-2,-1三个点是程序的输入节点
        '''
        node = Node(self.max_arity)
        node.i_func = random.randint(0, len(self.function_set) - 1)
        for i in range(self.function_set[node.i_func].arity):
            # 随机确定node的每个输入端口连接的是前面节点(column)的输出
            # node.i_inputs[i]记录前端父节点的idx
            node.i_inputs[i] = random.randint(max(pos - self.level_back, -self.n_inputs), pos - 1)
            node.weights[i] = 1.0 # random.uniform(self.weight_range[0], self.weight_range[1])
        node.i_output = pos

        return node

    def _determine_active_nodes(self):
        """
        Determine which nodes in the CGP graph are active
        """
        # check each node in reverse order
        n_active = 0

        # 逆序遍历所有节点
        for node in reversed(self.nodes):
            if node.active:
                n_active += 1
                # 依次检查该节点所有输入端口
                for i in range(self.function_set[node.i_func].arity):
                    # i_input是该node的第i的输入端口所连接的父节点的index
                    i_input = node.i_inputs[i]
                    if i_input >= 0:  # >=0表示node的父节点是一个节点，而非input点
                        # 该节点的父节点也设置为“激活”
                        self.nodes[i_input].active = True
        if config.Verbose:
            print("# active genes: ", n_active)

    def eval(self, *args):
        """
        Given inputs, evaluate the output of this CGP individual.
        :return the final output value
        """
        if not self._active_determined:
            self._determine_active_nodes()
            self._active_determined = True
        
        # forward pass: evaluate
        for node in self.nodes:
            if node.active:
                inputs = []
                for i in range(self.function_set[node.i_func].arity): # 依次获得该node各维输入
                    i_input = node.i_inputs[i] # node的父节点idx
                    w = node.weights[i] # node与父节点连边的权重
                    
                    # 父节点是程序的输入节点
                    if i_input < 0:
                        inputs.append(args[-i_input - 1] * w)
                    # 父节点是普通节点
                    else:
                        inputs.append(self.nodes[i_input].output * w)
                node.output = self.function_set[node.i_func](*inputs) # 执行计算
        if self.n_outputs == 1:
            return self.nodes[-1].output
        
        out = []
        for i in reversed(range(1, self.n_outputs + 1)):
            out.append(self.nodes[-i].output)
        return out

    def mutate(self, mut_rate=0.01):
        """
        Mutate this individual. Each gene is varied with probability *mut_rate*.
        :param mut_rate: mutation probability
        :return a child after mutation
        """
        child = copy.deepcopy(self)
        for pos, node in enumerate(child.nodes):
            # mutate the function gene
            if random.random() < mut_rate:
                node.i_func = random.choice(range(len(self.function_set)))
            
            # 函数突变之后需要重新选择其父节点
            # mutate the input genes (connection genes)
            arity = self.function_set[node.i_func].arity
            for i in range(arity):
                if node.i_inputs[i] is None or random.random() < mut_rate:  # if the mutated function requires more arguments, then the last ones are None 
                    node.i_inputs[i] = random.randint(max(pos - self.level_back, -self.n_inputs), pos - 1)
                if node.weights[i] is None or random.random() < mut_rate:
                    node.weights[i] = 1.0 # random.uniform(self.weight_range[0], self.weight_range[1])
            # initially an individual is not active except the last output node
            node.active = False
        for i in range(1, self.n_outputs + 1):
            child.nodes[-i].active = True
        child.fitness = None
        child._active_determined = False
        return child

Individual.function_set = fs
Individual.max_arity = max(f.arity for f in fs)

def evolve(pop, mut_rate, mu, lambda_):
    """
    Evolve the population *pop* using the mu + lambda evolutionary strategy

    :param pop: a list of individuals, whose size is mu + lambda. The first mu ones are previous parents.
    :param mut_rate: mutation rate
    :return: a new generation of individuals of the same size
    """
    pop = sorted(pop, key=lambda ind: ind.fitness)  # stable sorting
    parents = pop[-mu:]
    # generate lambda new children via mutation
    offspring = []
    for _ in range(lambda_):
        parent = random.choice(parents)
        offspring.append(parent.mutate(mut_rate))
    return parents + offspring


def create_population(n,input_dim,out_dim):
    """
    Create a random population composed of n individuals.
    """
    return [Individual(input_dim,out_dim) for _ in range(n)]



