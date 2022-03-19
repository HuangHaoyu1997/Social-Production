class config:
    seed = 1234
    Verbose = False              # 打印破产、解雇等事件
    render = True
    N = 1000                    # 智能体数量
    work_state = 0              # 初始工作状态, 0 unemployed, 1 employer, 2 worker
    init_coin = 100             # 初始货币
    random_coin = False          # 初始货币随机分布
    coin_range = [20,100]       # 初始货币随机分布区间,仅random_coin=True生效
    w1 = 10                     # 最低工资
    w2 = 90                     # 最高工资
    V = 1000                     # 初始市场价值
    T = 1000                    # 仿真步长
    avg_coin = (w1+w2)/2        # 平均工资
    avg_update = True           # 实时更新平均工资
    danjia = 1.0                # 工人向市场出售商品时，最大单价
    consume = 1.0               # 消费比例,即智能体每次消费占其总财产的比例
    die = False                 # 开启死亡机制，从agent_pool,E,W,U中删去
    hungry_days = 3             # 超过3天没钱消费就饿死
    x_range = [0,200]
    y_range = [0,200]
    skill = [0,1]               # 技能水平，实数
    resource = 5
    product_thr = 10            # 与资源最小距离＜10则会有产出
    move_len = 10               # 随机游走的最长距离
    move_dir = 1                # 随机游走的方向范围，即[0,2π]
    
    # for CGP
    MUT_PB = 0.7  # mutate probability
    N_COLS = 6   # number of cols (nodes) in a single-row CGP
    LEVEL_BACK = 4  # how many levels back are allowed for inputs in CGP
    # parameters of evolutionary strategy: MU+LAMBDA
    MU = 2
    LAMBDA = 100
    N_GEN = 50  # max number of generations


