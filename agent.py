import numpy as np

class agent:
    def __init__(self,param) -> None:
        
        self.name = param.name
        self.work = param.work_state
        self.coin = param.init_coin
        self.hire = {} # 雇佣工人集合
        self.employer = None


        
