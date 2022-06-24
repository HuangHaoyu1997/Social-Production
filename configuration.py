class config:
    '''
    w1,w2与init_coin的比例很大程度上决定了雇佣率的高低
    如果两者数量相当,则工人和失业者数量相近,工人略高
    如果w2远小于init_coin,则会出现少量资本家-少量失业者-大量工人的情况
    '''
    seed = 1234
    Verbose = False             # 打印破产、解雇等事件
    render = False
    print_log = False
    render_freq = 200           # 每100step保存figure
    N1 = 100                    # 初始人口
    N2 = 12000                  # 目标人口,即经过T步仿真后人口
    T = 500                     # 仿真步长
    Year = 100                  # 仿真100年

    # for agent
    work_state = 0              # 初始工作状态, 0 unemployed, 1 employer, 2 worker
    init_coin = 100             # 初始货币
    random_coin = True          # 初始货币随机分布
    coin_range = [50,150]       # 初始货币随机分布区间,仅random_coin=True生效
    employment_intention = 1.0  # 初始就业意愿
    move_len = 10               # 随机游走的最长距离
    move_dir = 1                # 随机游走的方向范围，即[0,2π]
    age = 15                    # 初始年龄
    age_mean = 25               # 初始群体的年龄呈现截断高斯分布,均值25岁,过高容易出现人口断崖下跌
    retire_age = 65             # 退休年龄, 超过退休年龄, 就业意愿就开始下降
    delta_age = Year / T        # 1 step = 100/1200 year = 1 month
    death_age = 75              # 年龄超过75岁,每个step都有25%的概率死亡
    death_prob = 0.15           # 自然死亡概率基准,每长1岁,加0.02

    # for government
    V = 100                     # 初始市场价值
    budget = 0                  # 初始政府财政
    dN = pow(N2/N1, 1/T) - 1 # 单步人口增量占当前人口比例,根据目标人口和初始人口进行推算
    w1 = 50                     # 初始最低工资
    w2 = 150                    # 初始最高工资
    salary_deque_maxlen = 1000  # 记录
    salary_gamma = 0.999        # discount factor,计算权重时使用,权重在时间上不是平均的,时间上越近,权重越高

    avg_salary = (w1+w2)/2        # 初始平均工资
    avg_update = True           # 实时更新平均工资
    # danjia = 1.0                # 工人向市场出售商品时，最大单价
    consume = 1.0               # 消费比例,即智能体每次消费量占其总财产的比例
    die = True                 # 开启死亡机制，从agent_pool,E,W,U中删去
    hungry_days = 3             # 超过3天没钱消费就饿死
    x_range = [0,200]
    y_range = [0,200]
    skill = [0,1]               # 技能水平，实数
    resource = 20
    product_thr = 10            # 与资源最小距离＜10则会有产出
    
    
    tax = True
    personal_income_tax = 0.0001 if tax else 0  # 个人所得税5%
    consumption_tax = 0.002 if tax else 0      # 消费税
    business_tax = 0.001 if tax else 0          # 企业税
    property_tax = 0.001 if tax else 0          # 财产税
    death_tax = 0.25 if tax else 0              # 遗产税
    redistribution_freq = 1    # 每10个step进行一次财富再分配
    property_tax_num = 50       # 财产税征收人数,即Top50富裕
    distribution_num = 100      # 政府税收再分配的人数,即Top100贫穷
    event_point = []     # 两次event,在point前后各持续2step
    event_duration = 1         # 负面事件的持续时间

    # for reward function
    target_RJ = 0.2             # 目标失业率小于10%


    # for CGP
    MUT_PB = 0.55  # mutate probability
    N_COLS = 50   # number of cols (nodes) in a single-row CGP
    LEVEL_BACK = 160  # how many levels back are allowed for inputs in CGP
    
    # parameters of evolutionary strategy: MU+LAMBDA
    MU = 2
    LAMBDA = 80-MU
    N_GEN = 5000  # max number of generations
    n_process = 40
    solved = 100
    Epoch = 10

    # for CGP visualization Postprocessing
    # if True, then the evolved math formula will be simplified and the corresponding
    # computational graph will be visualized into files under the `pp` directory
    PP_FORMULA = True
    PP_FORMULA_NUM_DIGITS = 5
    PP_FORMULA_SIMPLIFICATION = False
    PP_GRAPH_VISUALIZATION = True

    # for Policy Gradient Algorithm
    gamma = 0.99
    lr = 1e-2
    num_episodes = 200000000
    max_episode = 10 # test 10 episodes for one policy and average the total reward
    hidden_size = 128
    ckpt_freq = 10
    num_steps = 500
    batch = 30 # generate 20 tau for once time
    max_step = 500
    display = False
    epsilon = 0.4 # 一次生成batch个tau,根据reward选择前90%的样本进行训练


class CCMABM_Config:
    seed = 123
    x_range = [0,200]
    y_range = [0,200]
    skill = [0,1]               # 技能水平，实数
    tax = True
    base_tax_rate = 0.001       # 税率基准
    p_tax = 1.5                 # 个税倍率
    c_tax = 2.0                 # 消费税倍率
    
    
    T = 3000                    # number of simulation periods
    H = 3000                    # number of workers
    Fc = 200                    # number of C-firms
    Fk = 50                     # number of K-firms
    Ze = 5                      # number of firms visited by a unemployed worker
    Zc = 2                      # number of C-firms visited by a consumer
    Zk = 2                      # number of K-firms visited by a C-firm
    ksi = 0.96                  # Memory parameter (human wealth)
    tau = 0.2                   # Dividend payout ratio派息率
    chi = 0.05                  # Fraction of wealth devoted to consumption用于消费的财富比例
    r = 0.01                    # risk free interest rate无风险利率
    rho = 0.9                   # Quantity adjustment parameter数量调整参数
    import random
    eta = random.uniform(0, 0.1) # Price adjustment parameter (random variable)
    mu = 1.2                    # Bank's gross mark-up
    alpha = 0.5                 # Productivity of labor
    kappa = 1/3                 # Productivity of capital
    gamma = 0.25                # Probability of investing
    zeta = 0.002                # Bank's loss parameter
    theta = 0.05                # Installment on Debt 债务分期付款
    delta = 0.02                # Depreciation of capital 资本折旧
    nu = 0.5                    # Memory parameter (investment)
    omega_ = 0.85               # Desired capacity utilization rate 所需产能利用率
    w = 1                       # wage 工资
    Df1 = 10                    # Initial liquidity of (all) the firms所有公司的初始流动性
    K1 = 10                     # Initial capital初始资本
    Yc1 = 5                     # Initial production (C-firms)C公司初始产量
    Yk1 = 3                     # Initial production (K-firms)K公司初始产量
    Eb1 = 3000                  # Initial equity of the bank银行的初始抵押资产的净值
    Eh1 = 2                     # Initial households’ personal assets初始家庭个人资产