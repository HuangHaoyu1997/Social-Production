class config:
    N = 2000
    work_state = 0 # 0 unemployed, 1 employer, 2 worker
    init_coin = 1000
    w1 = 10 # 最低
    w2 = 90 # 最高工资
    V = 100 # 初始市场价值
    T = 1200 # 仿真步长
    avg_coin = (w1+w2)/2 # 平均工资
    danjia = 1.0 # 工人向市场出售商品时，最大单价

'''
con = config()
print(con.name)
'''
