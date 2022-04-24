import os
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
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

import torch
from DSO_Gym import REINFORCE, func_set, lstm
model_param = torch.load('./results/ckpt_CartPoleContinuous/reinforce-100.pkl')
model = lstm(input_size = 2*len(func_set),
                                hidden_size = 128, 
                                output_size = len(func_set), 
                                num_layer = 2
        )
model.load_state_dict(model_param)
agent = REINFORCE(func_set, 128)
agent.model = model
for _ in range(10):
    tau, _, _ = agent.symbolic_generator()
    print(tau)




