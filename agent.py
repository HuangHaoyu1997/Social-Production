import numpy as np

class agent:
    def __init__(self,param) -> None:
        
        self.name = param.name
        self.work = param.work_state # 工作状态
        self.coin = param.init_coin # 货币量
        self.hire = {} # 雇佣工人集合
        self.employer = None # 雇佣者


        
