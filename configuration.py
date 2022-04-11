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
    T = 200                     # 仿真步长
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
    
    
    tax = False
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
    target_RJ = 0.1             # 目标失业率小于10%


    # for CGP
    MUT_PB = 0.45  # mutate probability
    N_COLS = 100   # number of cols (nodes) in a single-row CGP
    LEVEL_BACK = 80  # how many levels back are allowed for inputs in CGP
    
    # parameters of evolutionary strategy: MU+LAMBDA
    MU = 2
    LAMBDA = 10-MU
    N_GEN = 5000  # max number of generations
    n_process = 20
    solved = 100
    Epoch = 2

    # for CGP visualization Postprocessing
    # if True, then the evolved math formula will be simplified and the corresponding
    # computational graph will be visualized into files under the `pp` directory
    PP_FORMULA = True
    PP_FORMULA_NUM_DIGITS = 5
    PP_FORMULA_SIMPLIFICATION = False
    PP_GRAPH_VISUALIZATION = True

