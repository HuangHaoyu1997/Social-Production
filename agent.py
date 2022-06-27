from random import uniform
import numpy as np
from configuration import config as c
from configuration import CCMABM_Config as CC
# import configuration as c
# np.random.seed(c.seed)

class agent:
    def __init__(self, position, name, skill, asset, age, intention) -> None:
        self.x, self.y = position
        
        self.skill = skill 
        self.alive = True
        self.hungry = 0
        self.name = name
        self.age = age
        self.employment_intention = intention # 就业意愿

        self.work = c.work_state # 工作状态
        self.asset = asset # 财产

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

class Firm:
    def __init__(self, 
                 ftype, 
                 init_capital,
                 init_deposit,
                 init_quantity,
                 wage,
                 rho, 
                 eta, 
                 alpha, 
                 kappa, 
                 gamma, 
                 delta, 
                 nu,
                 omega,
                 theta,
                 ) -> None:
        
        
        self.type = ftype # K-firm or C-firm
        
        self.t_price:float = None
        self.t_quantity:float = init_quantity
        self.tt_price:float = None
        self.tt_quantity:float = None
        
        self.labor_prod:float = alpha # 劳动生产率
        self.capital_prod:float = kappa # 资本生产率
        self.invest_prob:float = gamma # 当期投资概率
        self.capital_deprec:float = delta # 资本折旧率
        self.invest_memory:float = nu # 资本价格滑动平均参数
        self.desired_capacity_util:float = omega # 长期产能利用率
        self.installment:float = theta # 分期付款比例
        self.wage:float = wage
        
        self.avg_price = None
        self.capital = init_capital # 公司初始固定资产
        self.cap_t = None # t-1 step
        self.cap_tt = None
        self.deposit = init_deposit # 公司在银行的初始存款
        self.labor = None
        self.hire_list = []
        
        self.rho:float = rho # 产量调整参数
        self.eta:float = eta # 价格调整参数
        
    def avg_cap(self, ):
        '''
        calculating moving average capital used in past time 
        '''
        
        
    def production(self, ):
        
    
    def investment(self, ):
        pass
    
    def quantity_decision(self, avg_price, forecast_err):
        # forecast_err = 产量 - 销售量
        # forecast_err <0: 供小于求,应增加产量
        # forecast_err >0: 供大于求,应减少产量
        self.tt_quantity = self.t_quantity - self.rho * forecast_err
    
    def price_decision(self, avg_price, forecast_err):
        # 供小于求+低价-->涨价
        if forecast_err <= 0 and self.t_price < avg_price:
            self.tt_price = self.t_price * (1 + uniform(0, self.eta))
        # 供大于求+高价-->降价
        elif forecast_err > 0 and self.t_price >= avg_price:
            self.tt_price = self.t_price * (1 - uniform(0, self.eta))
        
        # 其他情况略微波动
        else:
            price_change = uniform(-self.eta*0.1, self.eta*0.1)
            self.tt_price = self.t_price * (1 + price_change)
    
    def decision(self, ):
        pass
    
    def sell(self, agent_list):
        '''sell productions to agent'''
        pass
    
    def fire(self, agent_list):
        '''fire a list of agents'''
        pass


class Bank:
    def __init__(self, ftype, 
                 init_capital, 
                 r,
                 mu,
                 zeta,
                 theta,
                 ) -> None:
        
        self.type = ftype
        self.equity = init_capital
        self.r_free = r
        self.r_risk = r * mu
        self.max_loss_ratio = zeta # 单笔贷款最大额度
        self.installment_debt = theta # 每期还款比例
        self.deposit_list = {}
        self.total_D = None
    
    def add_D(self, C_name, delta_m):
        assert delta_m>0 and C_name in self.deposit_list
        self.deposit_list[C_name] += delta_m
        return True
    
    def get_D(self, C_name):
        assert C_name in self.deposit_list
        return self.deposit_list[C_name]
    
    def withdraw_D(self, C_name, delta_m):
        assert delta_m>0 and C_name in self.deposit_list
        assert delta_m <= self.deposit_list[C_name]
        self.deposit_list[C_name] -= delta_m
        return True
    


class Government:
    def __init__(self, config:CC) -> None:
        self.config = config
        self._set_tax_rate(config)
        self.revenue = config.revenue
        
        
    def _set_tax_rate(self, c:CC):
        if c.tax: base_tax = c.base_tax_rate
        else: base_tax = 0.
        
        self.tax = {
            'base_tax': base_tax,
            'income': base_tax * c.p_tax,           # 个税
            'consumption': base_tax * c.c_tax,      # 消费税
            'business': base_tax * c.b_tax,         # 企业税
            'property': base_tax * c.pr_tax,        # 财产税
            'inheritance': base_tax * c.i_tax       # 遗产税
        }
        
    
    def taxing(self, money, tax_type):
        if isinstance(money, list):
            tax, surplus = [], []
            for mm in money:
                tax.append(mm*self.tax[tax_type])
                surplus.append(mm-tax[-1])
        else:
            tax = money * self.tax[tax_type]
            surplus = money - tax
        
        self.revenue += np.sum(tax)
        return surplus

class Market:
    def __init__(self, type, n_visit, ) -> None:
        self.type = type
        self.n_visit = n_visit
        self.goods_list = {}
        self.sold = {}
    def reset(self, ):
        self.goods_list = {}
    
    def add_goods(self, fname, production, price):
        self.goods_list[fname] = [production, price]
        
    def sell(self, cname, demand):
        pass
    
    def statistic(self, ):
        pass



class CCAgent(agent):
    def __init__(self, 
                 position, 
                 name, 
                 skill, 
                 asset, 
                 work, 
                 age, 
                 intention, 
                 memory,
                 chi,
                 ) -> None:
        super().__init__(position, name, skill, asset, age, intention)
        
        self.work = work # working type
        self.wage = None # 工资
        self.profit = None # 资本分红比例
        self.cash = None # 现金
        self.memory = memory # 计算期望收入的滑动平均系数
        self.consume_fraction = chi # 每期消费比例
        
        
        self.current_income = None
        self.history_income = None
    
    def consume_decision(self, ):
        pass
    
    def consume(self, firm_list:list):
        '''random search Zc C-firms for consumption'''
        pass
    
    def _get_wage(self, ):
        assert self.work != 'U'
        pass
    
    def deposit(self, bank:Bank):
        '''saving money into banks'''
        assert self.cash>0
        bank.add_D(self.name, self.cash)
    
    def set_demand(self, ):
        '''consumption decision'''
        pass
        
    def consume(self, market:Market):
        market.sell(self.name, )
    
    def employment(self, firm_list):
        '''random search Zd K- and C-firms for vacancies'''
        pass
    