import matplotlib.pyplot as plt
import numpy as np

sample = []
for i in range(1000):
    # sample.append(np.clip((np.random.randn()*10+38), 18, 100))
    sample.append(np.random.beta(4,4))


plt.figure()
plt.hist(sample, bins=100, density=0, facecolor="blue", edgecolor="black", alpha=0.7)

plt.xlabel("Area")
plt.ylabel("Number")
plt.show()
plt.clf()
