class config:
    seed = 123
    Verbose = False              # 打印破产、解雇等事件
    N = 1000                    # 智能体数量
    work_state = 0              # 0 unemployed, 1 employer, 2 worker
    init_coin = 100             # 初始货币
    random_coin = True          # 初始货币随机分布
    coin_range = [20,100]       # 初始货币随机分布区间,仅random_coin=True生效
    w1 = 10                     # 最低工资
    w2 = 90                     # 最高工资
    V = 100                     # 初始市场价值
    T = 100                    # 仿真步长
    avg_coin = (w1+w2)/2        # 平均工资
    avg_update = True           # 实时更新平均工资
    danjia = 1.0                # 工人向市场出售商品时，最大单价
    consume = 1.0               # 消费比例,即智能体每次消费占其总财产的比例
    die = False                 # 货币=0则死亡，从agent_pool,E,W,U中删去
    hungry_days = 3             # 超过3天没钱消费就饿死

