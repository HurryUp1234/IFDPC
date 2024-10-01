# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_process.data import specialized_IF
from data_process.data.specialized_IF import *


# 个体公平数据读取
def read_data_IF(data_name):
    if data_name == 'obesity':
        data_processed1 = get_data_obesity()
    elif data_name == 'bank':
        data_processed1 = get_data_bank()
    elif data_name == 'census1990':
        data_processed1 = get_data_census1990()
    elif data_name == 'creditcard':
        data_processed1 = get_data_creditcard()
    elif data_name == 'drug':
        data_processed1 = get_data_drug()
    elif data_name == 'dabestic':
        data_processed1 = get_data_dabestic()
    elif data_name == 'hcvdat0':
        data_processed1 = get_data_hcvdat0()
    elif data_name == 'athlete':
        data_processed1 = get_data_athlete()
    elif data_name == 'adult':
        data_processed1 = get_data_adult()
    elif data_name == 'drug_consumption':
        data_processed1 = get_data_drug_consumption()
    elif data_name == 'abalone':
        data_processed1 = get_data_abalone()
    elif data_name == 'Room_Occupancy_Estimation':
        data_processed1 = get_data_Room_Occupancy_Estimation()
    elif data_name == 'Rice':
        data_processed1 = get_data_Rice()
    elif data_name == 'Wholesale':
        data_processed1 = get_data_Wholesale()
    elif data_name == 'student':
        data_processed1 = get_data_student()
    elif data_name == 'parkinsons':
        data_processed1 = get_data_parkinsons()
    elif data_name == 'vertebral':
        data_processed1 = get_data_vertebral()
    elif data_name == 'liver_disorder':
        data_processed1 = get_data_liver_disorder()
    elif data_name == 'heart_failure_clinical':
        data_processed1 = get_data_heart_failure_clinical()
    elif data_name == 'chronic_kidney_disease':
        data_processed1 = get_data_chronic_kidney_disease()
    elif data_name == 'dermatology':
        data_processed1 = get_data_dermatology()
    elif data_name == 'glass':
        data_processed1 = get_data_glass()
    elif data_name == 'wdbc':
        data_processed1 = get_data_wdbc()
    elif data_name == 'wine':
        data_processed1 = get_data_wine()
    elif data_name == 'seeds':
        data_processed1 = get_data_seeds()
    else:
        data_processed1 = get_data_iris()

    data_processed = data_processed1
    need_subgraph = []
    # "abalone", "athlete", "Rice",
    # "adult", "census1990","Room_Occupancy_Estimation","creditcard"
    # 特定数据集的最佳阈值
    best_thresh = {
        # "abalone": 3.802,
        # "bank": 0.198,  # 0.17
        # "Rice": 0.155,
        # "Room_Occupancy_Estimation": 4.3,
        # "athlete": 0.09,  # 0.3: # 0.09:20%
        # "adult": 0.33,
        # "creditcard": 0.25,  # 0.3
        # "census1990": 5.68  # 8
    }

    # 创建一个默认值为1的字典
    thresholds = defaultdict(lambda: 1, best_thresh)

    # 查看数据集是否需要寻找最大联通子图
    if data_name in need_subgraph:
        print(f"need_subgraph: {data_name}, initial size: {len(data_processed)}")
        G = create_graph_from_data(data_processed, threshold=thresholds[data_name])
        selected_data = extract_largest_connected_subgraph(G, data_processed)
        print(f"after_subgraph size: {len(selected_data)}")
        # 保存处理后的数据
        output_path = f"{project_path}{data_name}_processed.csv"
        selected_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    else:
        selected_data = data_processed


    cluster_attributes_array = selected_data.values

    # 对聚类属性的数据进行归一化处理
    scaler = StandardScaler()
    cluster_attributes_normalized = scaler.fit_transform(cluster_attributes_array)

    # 敏感属性组-与对应点的编号
    return cluster_attributes_normalized