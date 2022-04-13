from utils import *

data = []
for i in np.linspace(-100,100,1000):
    data.append(tanh(i, alpha=0.05))
plt.figure()
plt.plot(np.linspace(-100,100,1000),data)
plt.show()