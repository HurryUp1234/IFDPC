import numpy as np

from FDPC import plot_FDPC, get_varience
from FDPC import FDPC2
from FDPC.FDPC3 import FDPC3
from FDPC.FDPC4 import FDPC4
from FDPC.FDPC5 import FDPC5
from FDPC.FDPC6 import FDPC6
from FDPC.get_varience import read_data, cal_dc_fair_loss
from mesure.comment import compute_SC, balance, AED_AWD
from sklearn import metrics
# 所有数据集都跑一遍

all_dataset = ['hcvdat0', 'liver_disorder', 'dermatology',
               'Wholesale', 'bank',
                'Rice', 'obesity', 'drug_consumption', 'adult']
# 'liver_disorder', 'dermatology', 'Wholesale','bank', 'hcvdat0', 'Rice', 'obesity', 'drug_consumption','adult',
all_dataset1 = ['abalone']
# 'abalone',"athlete","adult" "Rice",'bank','Room_Occupancy_Estimation'


# 需要最大联通子图的数据:
# abalone: 3.802 ,adult, athlete, bank: 0.05,
# Rice:0.155 , Room_Occupancy_Estimation: 20
# 有balance计算bug的数据：TamilSentiMix
for data_name in all_dataset1:
    print("")
    print("begin")
    print("data_name:" + data_name)
    dc = 0.05
    cluster_num = 4
    # [4, 0.01] # [8, 0.05] # [6, 0.03]
    cluster_attr_vals, f_attr_vals, f_attr = read_data(data_name)
    fdpc = FDPC6(cluster_attr_vals, f_attr_vals, f_attr, data_name,
                 dc_rate=dc, fair_dc_min_rate=dc / 2, fair_dc_max_rate=dc * 2,
                 cluster_num=cluster_num)
    # 对比实验，确保使用数据集，敏感，聚类属性相同
    dpc = FDPC2(data_name=data_name, dataset_cluster=fdpc.dataset_cluster, fair_attr_vals=fdpc.fair_attr_vals, fair_attr=fdpc.fair_attr,
                dc_rate=dc, cluster_num=cluster_num)
    print(f'f_dc: {fdpc.fair_dc},  dc: {fdpc.dc}')
    print(f'fair_loss_of_FDPC: {np.sum(cal_dc_fair_loss(fdpc, fdpc.fair_dc))}, '
          f'fair_loss_of_DPC: {np.sum(cal_dc_fair_loss(fdpc, fdpc.dc))}')
    plot_FDPC.draw_cluster(fdpc)
    plot_FDPC.draw_cluster(dpc)
    # balance
    fdpc_balance = balance(fdpc)
    dpc_balance = balance(dpc)
    balance_improvement = (fdpc_balance - dpc_balance) / dpc_balance
    # print('balance of FDPC:', fdpc_balance)
    # print('balance of DPC:', dpc_balance)
    # print(f'balance 提升率: {balance_improvement}')

    print('----------------------')
    # 轮廓系数
    fdpc_SC = compute_SC(fdpc)
    dpc_SC= compute_SC(dpc)
    # print(f'SC of FDPC: {fdpc_SC}')
    # print(f'SC of DPC: {dpc_SC}')
    SC_improvement = fdpc_SC - dpc_SC
    # print(f'SC 提升值: {SC_improvement}')