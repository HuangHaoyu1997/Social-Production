import networkx as nx
import matplotlib.pyplot as plt

plt.ion()

for i in range(10,30):
    plt.clf()
    G = nx.path_graph(i)
    subax1 = plt.subplot(111)
    nx.draw(G)
    plt.show()
    plt.pause(0.2)