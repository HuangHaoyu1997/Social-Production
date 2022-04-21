import os
import matplotlib.pyplot as plt

file_list = os.listdir('./results/exp6')
rewards = []
for file_name in file_list:
    # print(file_name.split('_')[0])
    rewards.append(float(file_name.split('_')[1]))


plt.plot(rewards)
plt.xlabel('episode'); plt.ylabel('reward')
plt.grid()
plt.show()



