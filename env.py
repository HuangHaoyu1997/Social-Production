import numpy as np
from configuration import config
import random
import math
import copy
from utils import *
import networkx as nx

random.seed(config.seed)
np.random.seed(config.seed)

class Env:
    def __init__(self) -> None:
        pass
        
    def reset(self,):
        self.t = 0 # tick
        self.agent_num = config.N # 智能体数量
        self.agent_pool = add_agent(self.agent_num) # 添加智能体
        self.market_V = config.V # 市场价值
        self.gov = config.G # 政府财政
        
        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态
        
        self.G = build_graph(self.agent_pool)
        self.resource = np.round(np.random.uniform(config.x_range[0],config.x_range[1],[config.resource,2]),2) # 资源坐标

        return [self.E, self.W ,self.U, self.market_V]
    
    def step(self,t,action):
        '''
        单步仿真程序

        '''
        if len(action)!=self.agent_num:
            raise Exception('动作空间必须与智能体数量相同')

        agent_list = list(self.agent_pool.keys())
        random.shuffle(agent_list)
        
        data_step = []
        
        for name in agent_list:
            # 必须活着
            if self.agent_pool[name].alive is not True: continue
            
            self.agent_pool[name].exploit = 0
            self.agent_pool[name].labor_cost = 0

            mod = round(random.uniform(0,config.move_len),2)
            dir = round(random.uniform(0,config.move_dir),3)
            self.agent_pool[name].move(mod=mod,direction=dir) # 移动
            
            self.production(name)
            self.hire(name)
            self.exploit(name)
            self.pay(name)
            data = self.consume(name)
            # self.fire(name)
            data_step.append(data)

        
        # 征收财产税
        rich_people = most_rich(self.agent_pool, 50)
        for name in rich_people:
            self.gov += self.agent_pool[name].coin * config.property_tax
        # 财富再分配
        if t % config.redistribution_freq == 0 and config.tax:
            self.redistribution()
        
        self.E, self.W, self.U = working_state(self.agent_pool)
        
        return data_step
        # 由于nx可视化的错误，暂时取消self.G的更新
        # self.G = update_graph(self.G,self.agent_pool,self.E,self.W,self.U)

    def redistribution(self, ):
        '''
        对政府的钱再分配
        '''
        # 【有可能出现总人数不足300的情况，按比例得出300这个数字会更好】
        poor_list = most_poor(self.agent_pool, 100)
        poor_num = len(poor_list)
        avg_coin = self.gov / poor_num
        for name in poor_list:
            self.agent_pool[name].coin += avg_coin
            self.gov -= avg_coin
        


    
    def production(self,name):
        '''
        产出量和距离成反比,和skill成正比
        '''
        shortest_dis = CalDistance(self.agent_pool[name], self.resource)
        if shortest_dis <= config.product_thr:
            product = (config.product_thr - shortest_dis)*round(random.uniform(0,self.agent_pool[name].skill),2)
            self.agent_pool[name].coin += product
            # print(name,shortest_dis,product)

    def hire(self, name):
        
        work = self.agent_pool[name].work
        
        # 【下一行能不能删去，安全吗？】
        self.E, self.W, self.U = working_state(self.agent_pool) # 可能耗费时间，确保都是活着的
        if work == 0: # 失业
            # U = copy.deepcopy(self.U); E = copy.deepcopy(self.E); random.shuffle(U)
            UE = self.U + self.E
            random.shuffle(UE)
            # name的潜在雇佣者的货币统计
            potential_e = [self.agent_pool[h].coin if self.agent_pool[h].coin>0 else 0 for h in UE]
            # print(potential_e)
            # 按货币量多少概率决定u的雇主
            prob = np.array(potential_e) / np.array(potential_e).sum()
            e = UE[np.random.choice(np.arange(len(prob)),p=prob)]
            if config.avg_update: 
                config.avg_coin = avg_coin(self.agent_pool,self.W+self.U)[0] # 更新平均工资
            if self.agent_pool[e].coin >= config.avg_coin and name!=e:
                # 给name设置雇主e,修改工作状态
                self.agent_pool[name].employer = e
                self.agent_pool[name].work = 2 # 2 for worker
                self.W.append(name)
                u_idx = self.U.index(name); self.U.pop(u_idx)
                # 给雇主e添加员工u,修改工作状态
                self.agent_pool[e].employer = e
                self.agent_pool[e].hire.append(name)
                self.agent_pool[e].work = 1 # 1 for employer
                if e not in self.E: self.E.append(e)
        elif work == 1: # 雇佣者
            pass
        elif work == 2: # 被雇佣者
            pass
        
        # 【这一行能不能删去，安全吗？】
        self.E, self.W, self.U = working_state(self.agent_pool)

    def exploit(self, name):
        work = self.agent_pool[name].work
        
        if work == 0: return None # 失业
        elif work == 1: employer = name # 雇主
        elif work == 2: employer = self.agent_pool[name].employer # 被雇佣

        w = np.random.uniform(low=0,high=int(config.danjia*self.market_V))
        w = round(w,2) # 四舍五入2位小数
        
        if config.tax:
            tax = w * config.business_tax
            w_ = w * (1-config.business_tax) # 扣除企业税
            self.gov += tax
            # print('ex,',tax)
        
        self.market_V -= w
        self.agent_pool[employer].coin += w_
        self.agent_pool[employer].exploit += w

    '''
        def pay_for_single(self, name):

        # 工作状态


        capital = self.agent_pool[name].coin # employer现有资本
        worker_list = self.agent_pool[name].hire # 雇佣名单
        random.shuffle(worker_list)
        for worker in worker_list:
            # 在最低工资和最高工资之间发工资
            w = np.random.uniform(config.w1, config.w2); w = np.round(w, 2)
            if capital >= w:
                self.agent_pool[worker].coin += w
                capital -= w
            # 资本量不足以开出现有工资
            elif capital < w and capital > config.w1:
                w = np.random.uniform(config.w1, capital); w = np.round(w, 2) # 降薪发工资
                self.agent_pool[worker].coin += w
                capital -= w
            # 连最低工资也开不出了
            elif capital <= config.w1:
                self.agent_pool[worker].coin += capital # 破产发工资
                capital = 0
                break
        
        if capital <= 0:
            self.broken(name) # 破产
            self.E, self.W, self.U = self.working_state() # 更新工作状态

    '''

    def pay(self, name):
        '''为agent_pool[name].hire中的worker发工资'''
        if self.agent_pool[name].work != 1: return None
        
        coin_before = self.agent_pool[name].coin # 发工资之前的资本
        
        if coin_before <= 0:
            self.broken(name) # 破产
        else:
            worker_list = self.agent_pool[name].hire # 雇佣名单
            random.shuffle(worker_list)
            for worker in worker_list:
                capital = self.agent_pool[name].coin # employer现有资本
                # 在最低工资和最高工资之间随机发工资
                w = np.random.uniform(config.w1, config.w2); w = round(w,2)
                if capital >= w:
                    capital -= w

                    # 缴个税
                    if config.tax:
                        tax = w * config.personal_income_tax
                        w = w * (1-config.personal_income_tax)
                        self.gov += tax
                        # print('pay,',tax)

                    self.agent_pool[worker].coin += w
                    self.agent_pool[name].coin = capital  ## 更新employer
                # 资本量不足以开出现有工资
                elif capital < w and capital > config.w1:
                    w = np.random.uniform(config.w1, capital); w = round(w,2) # 降薪发工资
                    capital -= w
                    
                    # 缴个税
                    if config.tax:
                        tax = w * config.personal_income_tax
                        w = w * (1-config.personal_income_tax)
                        self.gov += tax
                        # print('pay,',tax)
                    
                    self.agent_pool[worker].coin += w
                    self.agent_pool[name].coin = capital  ## 更新employer
                elif capital <= config.w1:
                    self.agent_pool[worker].coin += capital # 破产发工资
                    capital = 0
                    
                    # 低于最低工资，不交税
                    self.agent_pool[name].coin = capital  ## 更新employer
                    break
            coin_after = self.agent_pool[name].coin
        
            # 【保持configuration.py不变，程序运行628步后报错】
            if (coin_before - coin_after)<0: print(name, coin_before - coin_after)
            assert (coin_before - coin_after)>=0
        
            self.agent_pool[name].labor_cost = coin_before - coin_after
        
            # self.agent_pool[name].coin = capital # 【这一行有没有必要】
            if self.agent_pool[name].coin <= 0:
                self.broken(name) # 破产
        self.E, self.W, self.U = working_state(self.agent_pool) # 更新工作状态
        return None

    def broken(self,employer):
        '''
        雇佣者及其雇工都失业,加入U集合,从E和W集合中删除对应agent
        被雇者的雇主设置为0,修改其工作状态
        雇佣者的雇佣名单清空,修改其工作状态
        
        '''
        assert self.agent_pool[employer].work == 1
        if config.Verbose: print('%s has broken!'%employer)
        self.agent_pool[employer].work = 0 # 失业
        for worker in self.agent_pool[employer].hire:
            self.agent_pool[worker].work = 0 # 失业
            self.agent_pool[worker].employer = None
        self.agent_pool[employer].hire = []
        self.agent_pool[employer].employer = None
        self.agent_pool[employer].labor_cost = 0
        self.agent_pool[employer].exploit = 0
        return None

    def fire(self, name):
        '''
        解雇
        '''
        if self.agent_pool[name].work != 1: return None

        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态
        
        # 该资本家的货币量
        capital = self.agent_pool[name].coin
        # 雇工名单
        worker_list = self.agent_pool[name].hire
        # 最大工人数量=货币量/平均工资
        config.avg_coin = avg_coin(self.agent_pool, self.W+self.U)[0]
        max_num = np.floor(capital / config.avg_coin)
        # 工人数量
        num_worker = len(worker_list)
        fire_num = int(max(num_worker-max_num, 0)) # 解雇数量
        if fire_num > 0:
            fire_list = random.sample(self.agent_pool[name].hire, fire_num) # 随机解雇
            for worker in fire_list:
                # 改变工作状态
                assert worker in self.agent_pool[name].hire
                self.agent_pool[worker].work = 0
                self.agent_pool[worker].employer = None
                # 修改雇工名单
                fid = self.agent_pool[name].hire.index(worker)
                self.agent_pool[name].hire.pop(fid)
                if config.Verbose: print('%s has been fired by %s'%(worker,name))
        if len(self.agent_pool[name].hire) == 0: # 解雇所有雇员,自己也失业但没有破产
            self.agent_pool[name].work = 0
            self.agent_pool[name].employer = None
        
        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态

    def consume(self, name):
        # 消费，随机量m∈[0,m_a]
        # agent的钱减少m，市场价值增加m
        
        ma = self.agent_pool[name].coin
        data = [name,config.consume,ma,self.agent_pool[name].hungry]
        if ma > 0: 
            m = np.random.uniform(0, config.consume*ma)+1
            m_ = m
            # m = max(0.1*ma, 1)

            # 消费税
            if config.tax:
                tax = m * config.consumption_tax
                m_ = m * (1-config.consumption_tax)
                self.gov += tax
                # print('con,',tax)

            self.agent_pool[name].hungry = 0 # 要限制，连续饿三天才会死

        elif ma <= 0: 
            m = 0; m_ = 0
            self.agent_pool[name].hungry += 1

        self.agent_pool[name].coin -= m
        self.market_V += m_
        
        data.append(self.agent_pool[name].coin)
        data.append(self.market_V)
        data.append(self.agent_pool[name].hungry)

        if config.die: 
            if self.agent_pool[name].hungry >= config.hungry_days:
                self.die(name) # 饥饿超过5步就死掉
        
        return data
    
    def die(self, name):
        '''死亡'''
        assert self.agent_pool[name].coin <= 0
        # self.agent_pool.pop(name)
        if self.agent_pool[name].work == 1:
            self.broken(name)
        if self.agent_pool[name].work == 2:
            employer = self.agent_pool[name].employer
            idx = self.agent_pool[employer].hire.index(name)
            self.agent_pool[employer].hire.pop(idx)

        self.agent_pool[name].alive = False
        if name in self.E: idx = self.E.index(name); self.E.pop(idx)
        if name in self.U: idx = self.U.index(name); self.U.pop(idx)
        if name in self.W: idx = self.W.index(name); self.W.pop(idx)
        if config.Verbose: print('%s is dead!'%name)


# env.add_agent()
