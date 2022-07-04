from scipy.optimize import minimize
import numpy as np 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt 

m = 3.5
e = 1e-9
p = np.array([1.1, 1.04, 1.84])
q = np.array([1.13, 2.93, 0.69])

#目标函数：
def f1(x):
    f = x[0]/p[0] + x[1]/p[1] + x[2]/p[2]
    return -f
def f2(x):
    return -np.sum(x)

cons1 = ({'type': 'ineq', 'fun': lambda x: -x[0]/p[0]+q[0]},
        {'type': 'ineq', 'fun': lambda x: -x[1]/p[1]+q[1]},
        {'type': 'ineq', 'fun': lambda x: -x[2]/p[2]+q[2]},
        {'type': 'ineq', 'fun': lambda x: -x[0]-x[1]-x[2]+m})

cons2 = ({'type': 'ineq', 'fun': lambda x: -(x[0]-q[0])},
        {'type': 'ineq', 'fun': lambda x: -(x[1]-q[1])},
        {'type': 'ineq', 'fun': lambda x: -(x[2]-q[2])},
        {'type': 'ineq', 'fun': lambda x: x[0]-e},
        {'type': 'ineq', 'fun': lambda x: x[1]-e},
        {'type': 'ineq', 'fun': lambda x: x[2]-e},
        {'type': 'ineq', 'fun': lambda x: -(p[0]*x[0]+p[1]*x[1]+p[2]*x[2]-m)})
x0 = np.array((0,0,0)) #设置初始值，初始值的设置很重要，很容易收敛到另外的极值点中，建议多试几个值

#求解#
from time import time
t = time()
for _ in range(100):
    res = minimize(fun=f2,
               x0=x0,
               method='SLSQP',
               constraints=cons2, 
               options={'maxiter':5}
               ) # 'SLSQP'
print((time()-t)/100*1000)
# print(-res.fun)
# print(res.success)
solution = 3.5*res.x/np.sum(res.x)
print(res.x, res.x.sum())
# print(res.message)