from ast import Global
import multiprocessing
from multiprocessing import Process
import numpy as np
import time
import pickle


def func(idx, i,j):
    tmp = []
    for ii in range(i,j):
        tmp.append(ii)
        # print(tmp[ii])
    with open('./tmp/'+str(idx)+'.pkl','wb') as f:
        pickle.dump(tmp,f)
    print(idx,' finished!')

if __name__ == "__main__":
    '''
    
    '''
    tick = time.time()

    for i in range(10):
        p = Process(target=func, args=(i, i*1000000, (i+1)*1000000))
        p.start()
    p.join()
    print(time.time()-tick)
    tmp = []
    for i in range(10):
        with open('./tmp/'+str(i)+'.pkl','rb') as f:
            data = pickle.load(f)
            tmp.extend(data)
    print(np.mean(tmp))


    


