import numpy as np
from configuration import config as c
# import configuration as c
np.random.seed(c.seed)

class agent:
    def __init__(self, x, y, name, skill, coin) -> None:
        self.x = x
        self.y = y
        
        self.skill = skill 
        
        self.alive = True
        self.hungry = 0
        self.name = name
        self.work = c.work_state # 工作状态
        self.coin = coin # 货币量
        # self.coin = np.random.randint(low=50,high=100)
        self.hire = [] # 雇佣工人集合
        self.employer = None # 雇佣者
        self.throughput = 1.0 # 若agent为employer,该项生效,决定其worker的产量,未来应当作为agent的动作输出

        # special attributes for employers
        self.labor_cost = 0 # 人力成本 per timestep
        self.exploit = 0    # 剥削所得 per timestep
    
    def move(self,mod,direction):
        '''
        智能体移动
        mod:位移向量的模,[0,max_mod]
        direction:位移的方向,[0,1]表示0°~360°
        '''
        assert direction >= 0 and direction<=1
        assert mod >= 0
        assert self.alive

        dx = mod*np.cos(direction*2*np.pi)
        dy = mod*np.sin(direction*2*np.pi)
        
        self.x += dx
        self.y += dy

        if self.x < c.x_range[0]: self.x = c.x_range[0]
        elif self.x > c.x_range[1]: self.x = c.x_range[1]
        if self.y < c.y_range[0]: self.y = c.y_range[0]
        elif self.y > c.y_range[1]: self.y = c.y_range[1]

        
