class config:
    '''
    w1,w2与init_coin的比例很大程度上决定了雇佣率的高低
    如果两者数量相当,则工人和失业者数量相近,工人略高
    如果w2远小于init_coin,则会出现少量资本家-少量失业者-大量工人的情况
    '''
    seed = 123
    Verbose = False             # 打印破产、解雇等事件
    render = True
    N1 = 200                    # 初始人口
    N2 = 12000                   # 目标人口,即经过T步仿真后人口
    T = 10000                     # 仿真步长

    # for agent
    work_state = 0              # 初始工作状态, 0 unemployed, 1 employer, 2 worker
    init_coin = 100             # 初始货币
    random_coin = True          # 初始货币随机分布
    coin_range = [50,150]       # 初始货币随机分布区间,仅random_coin=True生效
    w1 = 10                     # 初始最低工资
    w2 = 90                     # 初始最高工资
    employment_intention = 1.0  # 就业意愿
    # for government
    V = 100                    # 初始市场价值
    G = 0                       # 初始政府财政
    dN = pow(N2/N1, 1/1000) - 1    # 单步人口增量占当前人口比例,根据目标人口和初始人口进行推算

    avg_coin = None # (w1+w2)/2        # 初始平均工资
    avg_update = True           # 实时更新平均工资
    # danjia = 1.0                # 工人向市场出售商品时，最大单价
    consume = 1.0               # 消费比例,即智能体每次消费量占其总财产的比例
    die = True                 # 开启死亡机制，从agent_pool,E,W,U中删去
    hungry_days = 3             # 超过3天没钱消费就饿死
    x_range = [0,200]
    y_range = [0,200]
    skill = [0,1]               # 技能水平，实数
    skill_gaussian = True       # 技能水平服从截断高斯分布
    resource = 10
    product_thr = 10            # 与资源最小距离＜10则会有产出
    move_len = 10               # 随机游走的最长距离
    move_dir = 1                 # 随机游走的方向范围，即[0,2π]
    
    tax = False
    personal_income_tax = 0.0001 if tax else 0  # 个人所得税5%
    consumption_tax = 0.002 if tax else 0      # 消费税
    business_tax = 0.001 if tax else 0          # 企业税
    property_tax = 0.001 if tax else 0          # 财产税
    redistribution_freq = 1    # 每10个step进行一次财富再分配
    event_duration = 30         # 负面事件的持续时间
    # for CGP
    MUT_PB = 0.45  # mutate probability
    N_COLS = 15   # number of cols (nodes) in a single-row CGP
    LEVEL_BACK = 80  # how many levels back are allowed for inputs in CGP
    
    # parameters of evolutionary strategy: MU+LAMBDA
    MU = 2
    LAMBDA = 20-MU
    N_GEN = 5000  # max number of generations
    n_process = 40
    solved = 100
    Epoch = 100

    # for CGP visualization Postprocessing
    # if True, then the evolved math formula will be simplified and the corresponding
    # computational graph will be visualized into files under the `pp` directory
    PP_FORMULA = True
    PP_FORMULA_NUM_DIGITS = 5
    PP_FORMULA_SIMPLIFICATION = True
    PP_GRAPH_VISUALIZATION = True

