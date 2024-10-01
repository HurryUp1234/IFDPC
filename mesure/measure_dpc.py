# -*- coding: utf-8 -*-

import numpy as np

from FDPC import plot_FDPC
from IFDPC.DPC import FDPC2
from FDPC.get_varience import read_data, cal_dc_fair_loss
from mesure.comment import compute_SC, balance

# 所有数据集都跑一遍

all_dataset = ['hcvdat0']
# 'liver_disorder', 'dermatology', 'Wholesale',
#                 'bank', 'hcvdat0', 'Rice', 'obesity', 'drug_consumption',
#                 'student', 'vertebral',
#                'adult',
all_dataset1 = ["obesity"]


# 淘汰的数据：heart_failure_clinical, parkinsons
# 良好可用的数据：
# 需要最大联通子图的数据:
# abalone: 3.802 ,adult, athlete, bank: 0.05, census1990, creditcard,
# Rice:0.155 , Room_Occupancy_Estimation: 20
# 读取数据有bug的数据：chronic_kidney_disease,   census1990(太大), creditcard
# 有balance计算bug的数据：TamilSentiMix
# dc为0的数据: drug
for data_name in all_dataset1:
    print("")
    print("")
    print("begin")
    print("")
    print("data_name:" + data_name)
    dc = 0.04
    cluster_num = 5
    # FDPC4：只看AED所有数据集都有提升
    cluster_attr_vals, f_attr_vals, f_attr = read_data(data_name)
    # 对比实验，确保使用数据集，敏感，聚类属性相同
    dpc = FDPC2(data_name, cluster_attr_vals, f_attr_vals, f_attr,
                dc)
    print(f'dc: {dpc.dc}')
    print(f'fair_loss_of_DPC: {np.sum(cal_dc_fair_loss(dpc, dpc.dc))}')
    plot_FDPC.draw_cluster(dpc)
    # balance
    dpc_balance = balance(dpc)
    print('balance of DPC:', dpc_balance)
    print('-----------------------')

    # 轮廓系数
    dpc_SC= compute_SC(dpc)
    print(f'SC of DPC: {dpc_SC}')