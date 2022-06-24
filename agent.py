import numpy as np
from configuration import config as c
from configuration import CCMABM_Config as CC
# import configuration as c
# np.random.seed(c.seed)

class agent:
    def __init__(self, x, y, name, skill, coin, age, intention) -> None:
        self.x = x
        self.y = y
        
        self.skill = skill 
        self.alive = True
        self.hungry = 0
        self.name = name
        self.age = age
        self.employment_intention = intention # 就业意愿

        self.work = c.work_state # 工作状态
        self.coin = coin # 货币量

        # special for employer
        self.hire = [] # 雇佣工人集合
        self.employer = None # 雇佣者
        self.throughput = 1.0 # 若agent为employer,该项生效,决定其worker的产量,未来应当作为agent的动作输出
        self.labor_cost = 0 # 人力成本 per timestep
        self.exploit = 0    # 剥削所得 per timestep
    
    def update_self_state(self,):
        '''
        目前的自身状态只有*就业意愿*一项
        就业意愿完全取决于self.age
        '''
        if self.age < 18:
            self.employment_intention = 0
        elif self.age >= 18 and self.age < c.retire_age:
            self.employment_intention = 1.0
        if self.age >= c.retire_age:
            self.employment_intention = max(self.employment_intention-0.02, 0.1)

    def move(self, mod, direction):
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

        
class CCAgent(agent):
    def __init__(self, x, y, name, skill, coin, work, age, intention) -> None:
        super().__init__(x, y, name, skill, coin, age, intention)
        self.work = work # working type
        self.wage = None # 工资
        self.profit = None # 资本分红比例
    
    def consume(self, firm_list):
        '''random search Zc C-firms for consumption'''
        pass
    
    def _get_wage(self, ):
        assert self.work != 'U'
        pass
    
    def deposit(self, bank):
        '''saving money into banks'''
        pass
    
    def employment(self, firm_list):
        '''random search Zd K- and C-firms for vacancies'''
        pass

class Government:
    def __init__(self, config:CC) -> None:
        self.config = config
        self._set_tax_rate(config)
        
    def _set_tax_rate(self, c:CC):
        if c.tax: self.base_tax = c.base_tax_rate
        else: self.base_tax = 0
        
        self.personal_income_tax = self.base_tax * c.p_tax  # 个人所得税5%
        self.consumption_tax = self.base_tax*     # 消费税
        self.business_tax = 0.001 if tax else 0          # 企业税
        self.property_tax = 0.001 if tax else 0          # 财产税
        self.death_tax = 0.25 if tax else 0              # 遗产税

class Firm:
    def __init__(self, ftype, init_capital) -> None:
        self.type = ftype # K-firm or C-firm
        self.current_price = None
        self.current_production = None
        self.avg_price = None
        self.capital = init_capital
        self.hire_list = []
    
    def production(self, ):
        pass
    
    def price(self, ):
        pass
    
    def investment(self, ):
        pass
    
    def decision(self, ):
        pass
    
    def sell(self, agent_list):
        '''sell productions to agent'''
        pass
    
    def fire(self, agent_list):
        '''fire a list of agents'''
        pass
    
class Bank(Firm):
    def __init__(self, ftype, init_capital) -> None:
        super().__init__(ftype, init_capital)
    

