import os, pickle
import matplotlib.pyplot as plt
import numpy as np

from utils import tSNE
'''
file_list = os.listdir('./results/exp6')
rewards = []
for file_name in file_list:
    # print(file_name.split('_')[0])
    rewards.append(float(file_name.split('_')[1]))


plt.plot(rewards)
plt.xlabel('episode'); plt.ylabel('reward')
plt.grid()
plt.show()

'''

'''
# \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
def Softplus(x, beta=1.0):
    return 1/beta * np.log(1+np.exp(beta*x))
y1, y2 = [], []
xx = np.linspace(-400,400,20000)
for x in xx:
    y1.append(Softplus(x,0.01))
    y2.append(Softplus(x,1.0))
plt.plot(xx,y1)
plt.plot(xx,y2)
plt.legend(['beta=0.01','beta=1'])
plt.grid()
plt.show()
'''

with open('./data/tSNE-simulation.pkl','rb') as f:
    data = pickle.load(f)
data = np.array(data)

data,_ = tSNE(data)
plt.figure(1)
plt.scatter(data[:,0],data[:,1], c=list(range(0, 501)))
plt.show()




