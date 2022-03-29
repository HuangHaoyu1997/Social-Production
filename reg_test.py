import numpy as np
import matplotlib.pyplot as plt 
def obj_fun(x):
    return x**2+2*x+1

def gen_data():
    y = []
    x = np.linspace(-1,1,200)
    for i in x:
        y.append(obj_fun(i))
    return x, np.array(y)

x,y = gen_data()
print(x,y)
