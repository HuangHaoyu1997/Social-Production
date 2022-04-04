import numpy as np
from numpy.random import uniform
from configuration import config
import random, math, copy
from utils import *
import networkx as nx

random.seed(config.seed)
np.random.seed(config.seed)

class Env:
    def __init__(self) -> None:
        pass
        
    def reset(self,):
        self.t = 0 # tick
        self.w1 = config.w1 # 初始最低工资
        self.w2 = config.w2 # 初始最高工资
        self.target_RJ = config.target_RJ # 失业率控制线


        # TODO【想法】开局不一定全部是unemployment，可以有UWE之分，每个人初始coin也可以不一样
        self.agent_pool = add_agent(config.N1) # 添加智能体
        self.market_V = config.V # 市场价值
        self.gov = config.budget # 政府财政
        
        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态
        
        self.G = build_graph(self.agent_pool)
        self.resource = np.round(uniform(config.x_range[0],config.x_range[1],[config.resource,2]),2) # 资源坐标

        self.w1_OU_noise = OrnsteinUhlenbeckActionNoise(mu=config.w1, theta=0.1, sigma=5, x0=10, dt=0.1) # config.w1
        info = self.ouput_info()
        return info
    
    def update_config(self, action=None):
        '''
        更新仿真超参数,用于实时控制系统
        '''
        if action is None:
            scale_factor = uniform(2,5)
            self.w1 = max(self.w1_OU_noise(), 1)
            self.w2 = scale_factor * self.w1
        else:
            self.w1 = action
            self.w2 = action*5

        # self.w1 = action[0]
        # self.w2 = action[1]
        # assert self.w1 < self.w2


    def step(self, action=None):
        '''
        单步仿真程序
        '''

        '''
        if len(action) != len(self.agent_pool):
            raise Exception('动作空间必须与智能体数量相同')
        '''
        self.update_config(action)

        agent_list = list(self.agent_pool.keys())
        random.shuffle(agent_list)
        
        ########### 单步执行
        # data_step = [] for policy symbolic regression
        for name in agent_list:
            # 必须活着
            if self.agent_pool[name].alive is not True: continue
            
            self.agent_pool[name].exploit = 0
            self.agent_pool[name].labor_cost = 0
            
            # 随机移动
            mod = round(uniform(0,config.move_len),2) # 向量的模
            dir = round(uniform(0,config.move_dir),3) # 向量的角度
            self.agent_pool[name].move(mod=mod,direction=dir)
            
            self.production(name)
            self.hire(name)
            self.exploit(name)
            self.pay(name)
            data = self.consume(name)
            self.fire(name)

            # data_step.append(data)

            # 更新年龄,更新自身状态,即就业意愿
            self.agent_pool[name].age += config.delta_age
            self.agent_pool[name].update_self_state()


            # 饿死或老死
            if config.die: 
                if self.agent_pool[name].hungry >= config.hungry_days:
                    self.die(name) # 饥饿超过5步就死掉
                else:
                    if self.agent_pool[name].age >= config.death_age \
                        and uniform()<=0.25+0.02*(self.agent_pool[name].age-config.death_age):
                        self.die(name)
            
            

        
        ########### 征收财产税
        rich_people = most_rich(self.agent_pool, 50)
        for name in rich_people:
            tax = self.agent_pool[name].coin * config.property_tax
            self.agent_pool[name].coin -= tax
            self.gov += tax

        ########### 财富再分配
        if config.tax and self.t % config.redistribution_freq == 0:
            self.redistribution()
        
        ########### 新增人口
        # 单步增加人口数量不超过当前人口的dN比例(0<dN<1)
        # TODO 目前增长速度正比于人口数量，应该开发logistic增长（S型增长曲线）
        delta_pop = np.random.randint( 0, max(round(config.dN*len(self.agent_pool)), 2) )
        self.agent_pool.update(add_agent(delta_pop))

        self.E, self.W, self.U = working_state(self.agent_pool)
        
        # 更新Graph
        self.G = update_graph(self.G, self.agent_pool, self.E, self.W, self.U)
        info = self.ouput_info()
        reward = self.reward_function(info)
        self.t += 1
        done = self.is_terminate()
        if config.print_log:
            if self.t % 12 == 0:
                print('t,\tem,\tun,\two,\talive,\tagt_c,\tmkt_c,\tttl_c,\tavg_u,\tavg_e,\tavg_w')
            print('%d,\t%d,\t%d,\t%d,\t%d,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t%.2f' % \
                (self.t, len(self.E),len(self.U),len(self.W),alive_num(self.agent_pool),\
                info['coin_a'],info['coin_v'],info['coin_t'],info['avg_coin_u'],info['avg_coin_e'],info['avg_coin_w'])
                )
        return info, reward, done
        
    def is_terminate(self,):
        if self.t >= config.T:
            return True
        elif alive_num(self.agent_pool) <= 0:
            return True
        else:
            return False
        
    def reward_function(self, info:dict):
        '''
        output: Reward Scalar
        '''
        reward_RJ = min(self.target_RJ - info['RJ'], 0) # 扣分项

        reward = reward_RJ
        return reward

    def ouput_info(self,):
        '''
        生成单步仿真的info数据,充当gym-like observation
        '''
        info = {}
        
        # 各部分人群的总财富
        coin_a, coin_v, coin_g, coin_t = total_value(self.agent_pool, self.market_V, self.gov)
        info['coin_a'] = coin_a
        info['coin_v'] = coin_v
        info['coin_g'] = coin_g
        info['coin_t'] = coin_t

        # 各部分人群的平均财富
        _, _, avg_coin_e, std_coin_e, _ = financial_statistics(self.agent_pool, self.E)
        _, _, avg_coin_w, std_coin_w, _ = financial_statistics(self.agent_pool, self.W)
        _, _, avg_coin_u, std_coin_u, _ = financial_statistics(self.agent_pool, self.U)
        _, _, avg_coin_t, std_coin_t, _ = financial_statistics(self.agent_pool, list(self.agent_pool.keys()))
        
        info['avg_coin_e'] = avg_coin_e; info['std_coin_e'] = std_coin_e
        info['avg_coin_w'] = avg_coin_w; info['std_coin_w'] = std_coin_w
        info['avg_coin_u'] = avg_coin_u; info['std_coin_u'] = std_coin_u
        info['avg_coin_t'] = avg_coin_t; info['std_coin_t'] = std_coin_t

        # 各部分人口
        info['Upop'] = len(self.U); info['Wpop'] = len(self.W); info['Epop'] = len(self.E); info['Tpop'] = alive_num(self.agent_pool)

        # 失业率Rate of Jobless
        info['RJ'] = info['Upop'] / info['Tpop']
        # 剩余价值率Rate of Surplus Value
        info['RSV'] = exploit_ratio(self.agent_pool, self.E)
        # 平均雇佣人数Rate of Hire
        info['RH'] = info['Wpop'] / info['Epop'] if info['Epop']>0 else 0
        # 最低/最高工资标准
        info['w1'] = self.w1
        info['w2'] = self.w2
        # 平均年龄
        info['avg_age'] = avg_age(self.agent_pool)
        

        return info # data_step
        

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
            product = (config.product_thr - shortest_dis)*round(uniform(0,self.agent_pool[name].skill),2)
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
                config.avg_coin = financial_statistics(self.agent_pool,self.W+self.U)[2] # 更新平均工资
            
            if self.agent_pool[e].coin >= config.avg_coin and name!=e:
                
                # TODO 考虑个体的就业意愿,100%
                if random.random() <= self.agent_pool[name].employment_intention:
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
        skill = self.agent_pool[name].skill
        if work == 0: return None # 失业
        elif work == 1: employer = name # 雇主
        elif work == 2: employer = self.agent_pool[name].employer # 被雇佣
        throughput = self.agent_pool[employer].throughput

        # 【若employer作为RL智能体, 则最低、最高价格应该由其控制,表示其能接受的毛利率上下限】
        w = round(uniform(0, skill*throughput*self.market_V) ,2) # 四舍五入2位小数

        tax = w * config.business_tax
        w_ = w * (1-config.business_tax) # 扣除企业税
        self.gov += tax
        # print('ex,',tax)
        
        self.market_V -= w
        self.agent_pool[employer].coin += w_
        self.agent_pool[employer].exploit += w

    def pay(self, name):
        '''
        为agent_pool[name].hire中的worker发工资
        '''
        if self.agent_pool[name].work != 1: return None
        
        coin_before = self.agent_pool[name].coin # 发工资之前的资本
        
        if coin_before <= 0: # 破产
            self.broken(name); print(f'{name} is broken before payment')
            
        else:
            worker_list = self.agent_pool[name].hire # 雇佣名单
            random.shuffle(worker_list)
            
            # TODO 通盘考虑所有工人，先在w1-w2之间开工资，如不行，整体降薪调整，如还不行，随机挑一个人减薪，工资考虑skill
            
            for worker in worker_list:
                # if not self.agent_pool[worker].alive: print(worker)
                # capital = self.agent_pool[name].coin # employer现有资本
                
                # 【情况1】在最低工资和最高工资之间随机发工资
                w = uniform(self.w1, self.w2)
                # print(name,w,self.agent_pool[name].coin)
                if self.agent_pool[name].coin >= w:
                    self.agent_pool[name].coin -= w
                    tax = w * config.personal_income_tax; self.gov += tax # 缴个税
                    w_after_tax = w * (1-config.personal_income_tax)
                    self.agent_pool[worker].coin += w_after_tax
                    continue
                    # self.agent_pool[name].coin = capital  ## 更新employer
                
                # 【情况2】资本量不足以开出现有工资，在最低工资和现有资本量之间开工资
                if self.agent_pool[name].coin < w and self.agent_pool[name].coin > self.w1:
                    w = uniform(self.w1, self.agent_pool[name].coin) # 降薪发工资
                    self.agent_pool[name].coin -= w
                    tax = w * config.personal_income_tax; self.gov += tax # 缴个人所得税
                    w_after_tax = w * (1-config.personal_income_tax)
                    self.agent_pool[worker].coin += w_after_tax
                    continue
                    # self.agent_pool[name].coin = capital  ## 更新employer
                
                # 【情况3】资本量少于最低工资，将全部资本用于发工资，然后破产
                if self.agent_pool[name].coin <= self.w1:
                    self.agent_pool[worker].coin += self.agent_pool[name].coin # 破产发工资
                    self.agent_pool[name].coin = 0
                    
                    # 低于最低工资，不交税
                    # self.agent_pool[name].coin = capital  ## 更新employer
                    break
            coin_after = self.agent_pool[name].coin
        
            # 【保持configuration.py不变，程序运行若干步后报错】
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
        id = self.E.index(employer)
        self.E.pop(id)

        for worker in self.agent_pool[employer].hire:
            self.agent_pool[worker].work = 0 # 失业
            self.agent_pool[worker].employer = None
            id = self.W.index(worker)
            self.W.pop(id)
        
        self.agent_pool[employer].hire = []
        self.agent_pool[employer].employer = None
        self.agent_pool[employer].labor_cost = 0
        self.agent_pool[employer].exploit = 0
        # return None

    def fire(self, name):
        '''
        解雇
        '''
        if self.agent_pool[name].work != 1: return None
        assert self.agent_pool[name].coin >= 0

        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态
        
        # 该资本家的货币量
        capital = self.agent_pool[name].coin
        # 雇工名单
        worker_list = self.agent_pool[name].hire
        # 最大工人数量=货币量/平均工资
        config.avg_coin = financial_statistics(self.agent_pool, self.W+self.U)[2]
        max_num = np.floor(capital / config.avg_coin)
        # max_num = np.ceil(capital / config.avg_coin)
        
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
        data = [name, config.consume, ma, self.agent_pool[name].hungry]
        if ma > 0: # 有钱就消费,消费要纳税
            # m = 0.1*ma
            m = uniform(0, config.consume*ma)
            
            tax = m * config.consumption_tax # 消费税
            m_ = m * (1-config.consumption_tax)
            self.gov += tax

            self.agent_pool[name].hungry = 0 # 要限制，连续饿三天才会死

        elif ma <= 0: # 没钱不消费,饥饿度+1
            m = 0; m_ = 0
            self.agent_pool[name].hungry += 1
        
        self.agent_pool[name].coin -= m
        self.market_V += m_

        # 资本家消费完了没钱了，也要破产
        if self.agent_pool[name].coin<=0 and self.agent_pool[name].work == 1:
            self.broken(name)

        data.append(self.agent_pool[name].coin)
        data.append(self.market_V)
        data.append(self.agent_pool[name].hungry)
        
        return data
    
    def die(self, name):
        '''
        死亡
        
        '''
        # assert self.agent_pool[name].coin <= 0
        # self.agent_pool.pop(name)
        # self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态
        if config.Verbose: print('%s is dead at %d years old with %f coins!'%(name, round(self.agent_pool[name].age), self.agent_pool[name].coin))
        
        # 资本家死前先破产
        if self.agent_pool[name].work == 1:
            self.broken(name)
        
        # 工人死前先失业
        if self.agent_pool[name].work == 2:
            self.jobless(name)
            # employer = self.agent_pool[name].employer
            # idx = self.agent_pool[employer].hire.index(name) # 从用工名单中删除
            # self.agent_pool[employer].hire.pop(idx)

        self.agent_pool[name].alive = False
        if name in self.E: idx = self.E.index(name); self.E.pop(idx)
        if name in self.U: idx = self.U.index(name); self.U.pop(idx)
        if name in self.W: idx = self.W.index(name); self.W.pop(idx)

        # 遗产继承
        if self.agent_pool[name].coin > 0 and len(self.E+self.W+self.U)>0:
            # print(alive_num(self.agent_pool),len(self.E+self.W+self.U),alive_num(self.agent_pool)==len(self.E+self.W+self.U))
            inheritor = random.sample(self.E+self.W+self.U, 1)[0] # 随机选继承人
            coin = self.agent_pool[name].coin
            tax = config.death_tax * coin; self.gov += tax # 缴税给政府
            coin_after_tax = coin * (1-config.death_tax)
            self.agent_pool[inheritor].coin += coin_after_tax # 继承人继承税后遗产
            self.agent_pool[name].coin = 0
        
    
    def jobless(self, worker):
        '''
        Worker失业函数
        强制修改Worker及其雇佣者的相关属性
        '''
        self.agent_pool[worker].work = 0
        e = self.agent_pool[worker].employer
        self.agent_pool[worker].employer = None

        id = self.agent_pool[e].hire.index(worker)
        self.agent_pool[e].hire.pop(id) # 从雇佣者的雇佣名单中除名
        
        if len(self.agent_pool[e].hire)==0: # 若Worker是其最后一个工人,则老板也失业
            self.broken(e)

    def event_simulator(self, event):
        '''
        异常事件发生器
        【想法:
        方案A(目前):event只持续1个step, 对agent的负面影响是即时生效的, 即大范围失业、贫困和死亡,
        方案B:让系统处于event较长一段时间, 如10~50个step, 每个step产生一定程度负面影响, 其总和等于方案A的影响
        哪种更好一些？】

        event目前设计2种事件: GreatDepression, WorldWar
        GreatDepression(大萧条):
            - E财富减少25%~50%, W财富减少25%~30%, U财富减少12%~25%,
            - E25%, W75%失业
            - U10%死亡
        WorldWar(世界大战):
            - E财富减少75%, W财富减少50%, U财富减少50%,
            - E25%~50%, W25%~50%失业
            - 全体10%~30%死亡
        PandemicInfluenza(大流感):
            - E死亡1%~3%, W死亡8%~10%, U死亡10%~12%
            - 全体失业1%~3%
            - 全体财富减少2%~5%
        '''
        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态

        if event == 'GreatDepression':
            if config.Verbose: print('GreatDepression is coming!')
            
            # 财产损失
            for name in self.agent_pool:
                if self.agent_pool[name].work == 0:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.02,0.05) # 0.12,0.25
                elif self.agent_pool[name].work == 1:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.03,0.07) # 0.25,0.50
                elif self.agent_pool[name].work == 2:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.04,0.09) # 0.25,0.30
            
            # 破产
            broken_rate = uniform(0.1, 0.2)
            broken_E = random.sample(self.E, int(broken_rate*len(self.E))) # 25%的资本家破产
            for name in broken_E: # 破产即失业
                self.broken(name)
            
            jobless_rate = uniform(0.2, 0.3)
            broken_W = random.sample(self.W, int(jobless_rate*len(self.W))) # 25%的工人失业
            for name in broken_W:
                if self.agent_pool[name].work == 2: # 还没失业，就让他失业
                    self.jobless(name)
            
            # 死亡
            death_rate = uniform(0.03, 0.05)
            dead_U = random.sample(self.U, int(death_rate*len(self.U))) # 10%的失业者死亡
            for name in dead_U:
                if self.agent_pool[name].alive:
                    self.die(name)
        
        elif event == 'WorldWar':
            if config.Verbose: print('WorldWar is coming!')
            for name in self.agent_pool:
                if self.agent_pool[name].work == 0:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.40,0.50)
                elif self.agent_pool[name].work == 1:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.65,0.75)
                elif self.agent_pool[name].work == 2:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.30,0.50)
            
            broken_ratio = uniform(0.25,0.50)
            broken_E = random.sample(self.E, int(broken_ratio*len(self.E))) # 25%~50%的资本家破产
            for name in broken_E: # 破产即失业
                self.broken(name)

            broken_W = random.sample(self.W, int(broken_ratio*len(self.W))) # 25%~50%的工人失业
            for name in broken_W:
                if self.agent_pool[name].work == 2: # 还没失业，就让他失业
                    self.jobless(name)
            
            dead_ratio = uniform(0.10,0.30)
            dead_U = random.sample(self.U+self.E+self.W, int(dead_ratio*len(self.U))) # 10%~30%的人死亡
            for name in dead_U:
                if self.agent_pool[name].alive:
                    self.die(name)

        elif event == 'PandemicInfluenza':
            if config.Verbose: print('Pandemic Influenza is coming!')
            for name in self.agent_pool:
                self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.02,0.05)
                '''
                if self.agent_pool[name].work == 0:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.02,0.05)
                elif self.agent_pool[name].work == 1:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.02,0.05)
                elif self.agent_pool[name].work == 2:
                    self.agent_pool[name].coin -= self.agent_pool[name].coin*uniform(0.02,0.05)
                '''
                
            broken_ratio = uniform(0.01,0.03)
            broken_A = random.sample(list(self.agent_pool.keys()), int(broken_ratio*len(self.agent_pool)))
            for name in broken_A:
                self.broken(name)
            
            dead_ratio = uniform(0.01,0.02)
            dead_E = random.sample(self.E, int(dead_ratio*len(self.E))) # 10%~30%的人死亡
            for name in dead_E:
                if self.agent_pool[name].alive: self.die(name)
            
            dead_ratio = uniform(0.08,0.10)
            dead_W = random.sample(self.W, int(dead_ratio*len(self.W))) # 10%~30%的人死亡
            for name in dead_W:
                if self.agent_pool[name].alive: self.die(name)
            
            dead_ratio = uniform(0.10,0.12)
            dead_U = random.sample(self.U, int(dead_ratio*len(self.U))) # 10%~30%的人死亡
            for name in dead_U:
                if self.agent_pool[name].alive: self.die(name)
        
        else:
            raise NotImplementedError
        self.E, self.W, self.U = working_state(self.agent_pool) # 更新维护智能体状态


# env.add_agent()
