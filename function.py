import numpy as np
import math
import operator as op


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

def protected_div(a, b):
    if abs(b) < 1e-6:
        return 1 # a / (b+1e-6)
    else:
        return a / b

def sqrt(a):
    return math.sqrt(abs(a))

def relu(a):
    if isinstance(a,complex): print(a)
    if a>=0: return a
    else: return 0

def ln(a):
    if a>=-0.001 and a<=0.001:
        return 0
    else:
        return math.log(abs(a))

def exp(a):
    return (np.exp(a)-1)/(np.exp(1)-1)

def max1(a):
    return max(a,0)

def max2(a,b):
    if a <= b: return b
    else: return a

def min2(a,b):
    if a <= b: return a
    else: return b

def tenth(a):
    return a*0.1

def scaled(a):
    # 压缩到[-1,1]区间
    if a is None: return 0.0
    return min(max(a, -1.0), 1.0)

def abs(a):
    return np.abs(a)

def pi(a):
    return a*np.pi

def sign(a):
    '''
    1-sign(x)
    '''
    if a>0: return 0
    else: return 1

def sin(a):
    return np.sin(a)

def inv(a):
    if a>=-0.001 and a<=0.001:
        return 1
    else:
        return 1/a

def uniform(a):
    if a<=0: return 0
    return np.random.uniform(0,a)

fs = [
        Function(op.add, 2), 
        Function(op.sub, 2), 
        Function(op.mul, 2), 
        # Function(protected_div, 2), 
        # Function(op.neg, 1),
        # Function(op.pow, 2),
        # Function(exp, 1),
        # Function(max2, 2),
        Function(max1, 1),
        # Function(min2, 2),
        Function(tenth, 1),
        # Function(scaled, 1),
        Function(sign, 1),
        # Function(uniform, 1),
        # Function(relu, 1),
        # Function(abs, 1),
        # Function(sin, 1),
        # Function(pi,1)
        # Function(ln, 1),
        # Function(sqrt, 1),
        # Function(inv, 1),
    ]