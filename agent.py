import numpy as np
from configuration import config as c
# import configuration as c
np.random.seed(c.seed)

class agent:
    def __init__(self,name) -> None:
        self.x = round(np.random.uniform(c.x_range[0],c.x_range[1]),2)
        self.y = round(np.random.uniform(c.y_range[0],c.y_range[1]),2)
        self.skill = round(np.random.uniform(c.skill[0],c.skill[1]),2)
        
        self.alive = True
        self.hungry = 0
        self.name = name
        self.work = c.work_state # 工作状态
        self.coin = np.random.uniform(c.coin_range[0],c.coin_range[1]) if c.random_coin else c.init_coin # 货币量
        # self.coin = np.random.randint(low=50,high=100)
        self.hire = [] # 雇佣工人集合
        self.employer = None # 雇佣者


        
