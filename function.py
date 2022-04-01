import numpy as np
import math
import operator as op
import sympy as sp

class Function:
    """
    A general function
    arity: 函数的输入参数的数量
    """

    def __init__(self, f, arity, name=None):
        self.f = f
        self.arity = arity
        self.name = f.__name__ if name is None else name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def protected_div(a, b, epsilon=1e-3):
    if abs(b) < epsilon:
        return a / (b+epsilon)
    else:
        return a / b

def sqrt(a):
    '''
    return: sqrt(abs(a))
    '''
    return math.sqrt(abs(a))

def relu(a):
    if isinstance(a,complex): print(a)
    if a>=0: return a
    else: return 0

def ln(a, epsilon=1e-3):
    if abs(a) <= epsilon:
        return 0
    else:
        return math.log(abs(a))

def exp(a):
    return (np.exp(a)-1)/(np.exp(1)-1)

def max1(a):
    # print(a)
    return max(a,0) # ,sp.maximum(a,sp.sqrt(1))

def min1(a):
    '''
    min1是冗余的,可以用op.neg+max1来实现
    '''
    return min(a,0)

def max2(a,b):
    if a <= b: return b
    else: return a

def min2(a,b):
    if a <= b: return a
    else: return b

def tenth(a):
    return a*0.1

def scaled(a):
    '''
    压缩到[-1,1]区间
    '''
    if a is None: return 0.0
    return min(max(a, -1.0), 1.0)

def pi(a):
    return a*np.pi

def sign(a):
    '''
    其实是1-sign(x)
    '''
    if a>0: return 0
    else: return 1

def sin(a):
    return np.sin(a)

def inv(a, epsilon=1e-3):
    if abs(a) <= epsilon:
        return 1
    else:
        return 1/a

def const_1():
    '''常数1.0'''
    return 1.0

def const_5():
    '''常数5.0'''
    return 5.0

def const_tenth():
    '''常数0.1'''
    return 0.1

def unif():
    '''
    [0,1]均匀分布
    '''
    return np.random.uniform(0,1)

def uniform(a):
    if a<=0: return 0
    return np.random.uniform(0,a)

fs = [
        Function(const_1, 0),
        # Function(const_5, 0),
        # Function(const_tenth, 0),

        Function(op.add, 2), 
        # Function(op.sub, 2), 
        Function(op.mul, 2), 
        # Function(protected_div, 2),
        # Function(op.neg, 1),
        # Function(op.abs, 1),
        # Function(op.ge, 2),
        # Function(op.le, 2),
        
        # Function(op.pow, 2),
        # Function(exp, 1),
        # Function(max1, 1),
        # Function(min1, 1),
        # Function(max2, 2),
        # Function(min2, 2),
        # Function(tenth, 1),
        # Function(scaled, 1),
        # Function(sign, 1),
        # Function(uniform, 1),
        # Function(relu, 1),
        # Function(sin, 1),
        # Function(pi,1)
        # Function(ln, 1),
        # Function(sqrt, 1),
        # Function(inv, 1),
    ]
fs = [
        Function(op.add, 2), 
        Function(op.sub, 2), 
        Function(op.mul, 2), 
        # Function(protected_div, 2), 
        Function(op.neg, 1),
        # Function(op.pow, 2),
        # Function(exp, 1),
        # Function(max2, 2),
        Function(max1, 1),
        # Function(min2, 2),
        Function(tenth, 1),
        Function(scaled, 1),
        Function(sign, 1),
        # Function(uniform, 1),
        # Function(relu, 1),
        Function(abs, 1),
        # Function(sin, 1),
        # Function(pi,1)
        # Function(ln, 1),
        # Function(sqrt, 1),
        # Function(inv, 1),
    ]
# Map Python functions to sympy counterparts for symbolic simplification.
DEFAULT_SYMBOLIC_FUNCTION_MAP = {
    # op.and_.__name__:       sp.And,
    # op.or_.__name__:        sp.Or,
    # op.not_.__name__:       sp.Not,
    op.add.__name__:        op.add,
    op.sub.__name__:        op.sub,
    op.mul.__name__:        op.mul,
    op.neg.__name__:        op.neg,
    'max1':                 max1,
    'tenth':                tenth,
    'scaled':               scaled,
    'sign':                 sign,
    'abs':                  op.abs,
    # op.pow.__name__:        op.pow,
    # op.abs.__name__:        op.abs,
    # op.floordiv.__name__:   op.floordiv,
    # op.truediv.__name__:    op.truediv,
    # 'protected_div':        op.truediv,
    # math.log.__name__:      sp.log,
    # math.sin.__name__:      sp.sin,
    # math.cos.__name__:      sp.cos,
    # math.tan.__name__:      sp.tan,
    # 'const_1':              const_1,
    # 'const_5':              const_5,
    # 'const_tenth':          const_tenth,

}

if __name__ == '__main__':
    
    pass