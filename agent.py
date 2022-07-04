from random import sample, uniform
import numpy as np
import random
from configuration import config as c
from configuration import CCMABM_Config as CC
# import configuration as c
# np.random.seed(c.seed)
import warnings
warnings.filterwarnings("ignore")
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
                 name,
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
        
        self.step = 0
        self.name = name
        self.type:str = ftype # K-firm or C-firm
        
        self.t_price:float = None
        self.t_quantity:float = init_quantity
        self.tt_price:float = None      # t+1 step价格
        self.tt_quantity:float = None   # t+1 step产量
        
        self.labor_prod:float = alpha   # 劳动生产率
        self.capital_prod:float = kappa # 资本生产率
        self.invest_prob:float = gamma  # 当期投资概率
        self.capital_deprec:float = delta # 资本折旧率
        self.invest_memory:float = nu   # 资本价格滑动平均参数
        self.desired_capacity_util:float = omega # 长期产能利用率
        self.installment:float = theta  # 分期付款比例
        self.wage:float = wage
        
        self.avg_price = None
        self.capital = init_capital # 公司初始固定资产
        self.cap_t = init_capital # t-1 step
        self.cap_tt = init_capital # t-2 step
        self.cap_avg = [init_capital]*2
        
        self.deposit = init_deposit # 公司在银行的初始存款
        self.labor = None
        self.hire_list = []
        
        self.rho:float = rho # 产量调整参数
        self.eta:float = eta # 价格调整参数
        
    def avg_cap(self, ):
        '''
        calculating moving average capital used in past time 
        '''
        self.cap_avg
        
    def get_production(self, ):
        
        return self.t_quantity, self.t_price
    
    def investment(self, ):
        pass
    
    def quantity_decision(self, avg_price, forecast_err):
        
        # forecast_err = 产量 - 销售量
        # forecast_err <0: 供小于求,高于均价->增加产量
        # forecast_err >0: 供大于求,低于均价->减少产量
        # 其他情况小范围随机波动
        if forecast_err<0 and self.t_price >= avg_price:
            self.tt_quantity = self.t_quantity - self.rho * forecast_err
        elif forecast_err>0 and self.t_price < avg_price:
            self.tt_quantity = self.t_quantity - self.rho * forecast_err
        else:
            self.tt_quantity = self.t_quantity + uniform(-0.1*self.rho, 0.1*self.rho)
    
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
    
    def get(self, agent_list):
        '''sell productions to agent'''
        pass
    
    def fire(self, number):
        '''fire a list of agents'''
        fire_list = random.sample(self.hire_list, number)
        assert len(fire_list) < len(self.hire_list)

        for fname in fire_list:
            fid = self.hire_list.index(fname)
            self.hire_list.pop(fid)
        return fire_list
    
    def pay_wage(self, ):
        '''pay the salary to workers hired in this firm'''
        assert len(self.hire_list)>0
        wage_list = {}
        for name in self.hire_list:
            wage_list[name] = self.wage
            self.deposit -= self.wage
        
        return wage_list


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
    
    def add_goods(self, fname, quantity, price):
        self.goods_list[fname] = [quantity, price]
    
    def sort(self, ):
        pass
    
    def add_sold(self, fname, cname, quantity, price):
        if fname not in self.sold:
            self.sold[fname] = []
        
        self.sold[fname].append({cname: [price, quantity]})
    
    def sell(self, name, m_demand):
        '''
        name: consumer name
        m_demand: money for consumption
        '''
        # consumers only visit a part of firms
        fname_sampled = random.sample(self.goods_list.keys(), 
                                      self.n_visit)
        
        # 按价格从低到高排序sort the goods by price
        good_quantity = np.array([self.goods_list[f][0] for f in fname_sampled])
        good_price = np.array([self.goods_list[f][1] for f in fname_sampled])
        
        # idx = np.argsort(good_price) # less --> more
        # good_quantity_sorted = good_quantity[idx]
        # good_price_sorted = good_price[idx]
        # fname_sampled_sorted = np.array(fname_sampled)[idx]
        
        # from sko.PSO import PSO
        # def object(money):
        #     return -np.sum([m/p for m,p in zip(money, good_price)])
        # constraint_ueq = (
        #     lambda m: m[0]/good_price[0]-good_quantity[0], # n_visit
        #     lambda m: m[1]/good_price[1]-good_quantity[1],
        #     lambda m: m[2]/good_price[2]-good_quantity[2],
        # )
        # pso = PSO(func=object, 
        #           n_dim=self.n_visit, 
        #           pop=40, 
        #           max_iter=200, 
        #           lb=0, 
        #           ub=m_demand, 
        #           constraint_ueq=constraint_ueq)
        # pso.run()
        # # 消费金额的最优分配方案
        # quan = pso.gbest_x / good_price
        
        def consumption_solve(p, q, m, n_visit):
            from scipy.optimize import minimize
            f = lambda x: -np.sum(x)
            e = 1e-10
            cons = ({'type': 'ineq', 'fun': lambda q_: q[0]-q_[0]},
                    {'type': 'ineq', 'fun': lambda q_: q[1]-q_[1]},
                    {'type': 'ineq', 'fun': lambda q_: q[2]-q_[2]},
                    {'type': 'ineq', 'fun': lambda q_: q_[0]-e},
                    {'type': 'ineq', 'fun': lambda q_: q_[1]-e},
                    {'type': 'ineq', 'fun': lambda q_: q_[2]-e},
                    {'type': 'ineq', 'fun': lambda q_: m-(p[0]*q_[0]+p[1]*q_[1]+p[2]*q_[2])})
            x0 = np.array((m, m, m))/n_visit #设置初始值，初始值的设置很重要，很容易收敛到另外的极值点中，建议多试几个值
            res = minimize(fun=f,
                        x0=x0,
                        method='SLSQP',
                        constraints=cons, 
                        options={'maxiter':10}
                        )
            quantity_decision = np.round(res.x, 2)
            return quantity_decision
        
        quantity_decision = consumption_solve(p=good_price,
                                              q=good_quantity,
                                              m=m_demand,
                                              n_visit=self.n_visit)
        
        # 消费基金的剩余, 返还消费者
        surplus = m_demand - sum(quantity_decision*good_price)
        surplus = surplus if surplus>0 else 0
        
        for fname, q_, p, q in zip(fname_sampled, quantity_decision, good_price, good_quantity):
            # 记录成交订单
            if round(q_, 2)>0:
                self.add_sold(fname=fname, # 商家名
                              cname=name, # 消费者名
                              quantity=round(q_, 2), # 消费数量
                              price=round(p, 2)) # 价格
            
            # 更新库存量=库存量q - 消费量q_
            if round(q - q_, 2)==0: # 售罄,清空good_list
                self.goods_list.pop(fname)
                continue
            self.goods_list[fname][0] = round(q - q_, 2)
            
        return surplus, sum(quantity_decision) # 剩下的钱, 消费量
    
    def statistic(self, ):
        firm_sold = {}
        
        for fname in self.sold:
            sold_quanity = 0
            for cname in self.sold[fname]:
                sold_quanity += cname[list(cname.keys())[0]][1]
            print(fname, round(sold_quanity, 2))



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
    
    def consume(self, good_dict:dict):
        '''random search Zc C-firms for consumption'''
        for name in good_dict:
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
    
if __name__ == '__main__':
    m = Market(type='C', n_visit=3)
    for t in range(100):
        m.add_goods(fname=str(t), 
                    price=round(uniform(1,5), 2),
                    quantity=round(uniform(10,30), 2))
    for i in range(500):
        m.sell(name='C'+str(i), m_demand=uniform(3, 10))
    m.statistic()