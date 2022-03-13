import numpy as np
from config import *

np.random.seed(config.seed)

class agent:
    def __init__(self,param) -> None:
        self.alive = True
        self.hungry = 0
        self.name = param.name
        self.work = param.work_state # 工作状态
        self.coin = np.random.randint(low=config.coin_range[0],high=config.coin_range[1]) if config.random_coin else param.init_coin # 货币量
        # self.coin = np.random.randint(low=50,high=100)
        self.hire = [] # 雇佣工人集合
        self.employer = None # 雇佣者


        
