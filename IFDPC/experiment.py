# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from openpyxl import Workbook

from IFDPC import get_varience
from IFDPC.DPC import DPC
from IFDPC.DPC_var.KS_FDPC_2_CCFB import KS_FDPC
from IFDPC.DPC_var.SFKNN_DPC__CCFB import SFKNN_DPC
from IFDPC.IFDPC4 import IFDPC4
from IFDPC.IFDPC5 import IFDPC5
from IFDPC.comment import *




def create_excel_file(data_name):


    df = pd.DataFrame(columns=['聚类数', 'K数', 'DPC方差比准则', 'IFDPC方差比准则', 'SFKNN-DPC方差比准则', 'KS-FDPC方差比准则', 'DPC个体公平性', 'IFDPC个体公平性',
                               'IFDPC个体公平性提升', 'SFKNN-DPC个体公平性', 'SFKNN-DPC个体公平性提升', 'KS-FDPC个体公平性', 'KS-FDPC个体公平性提升'])

    # Save it as an Excel file
    file_name = f"{data_name}_results.xlsx"
    df.to_excel(file_name, index=False)
    return file_name


# Function to append results to the corresponding Excel file
def append_to_excel(file_name, results):
    # Load the Excel file
    df = pd.read_excel(file_name)
    # Append new row
    df = df._append(results, ignore_index=True)
    # Save back to the same file
    df.to_excel(file_name, index=False)


if __name__ == '__main__':

    # 参数
    all_datasets = [
         'Wholesale',
        'obesity',
        #'hcvdat0',
        'athlete',
                    'abalone',
                    'seeds',
                    'bank',
                    # 'census1990',
                    # 'creditcard',
                   #  'adult',
        # 'Rice'
                   ]


    cluster_range = range(4, 9)
    K_range = range(6, 11)
    for data_name in all_datasets:
        data = get_varience.read_data_IF(data_name)
        file_name = create_excel_file(data_name)

        for cluster_num in cluster_range:
            for K in K_range:
                # IFDPC = IFDPC5(data, data_name, K, cluster_num=cluster_num)
                DPC1 = DPC(data_name, data, cluster_num=cluster_num)

                #SFKNN_DPC1 = SFKNN_DPC(data, K, cluster_num)
                #SFKNN_DPC1.fit()
                KS_FDPC1 = KS_FDPC(data, K, cluster_num)
                KS_FDPC1.fit()
                # 轮廓系数
                DPC_SC = compute_SC(DPC1)
                IFDPC_SC = 1 #compute_SC(IFDPC)
                SFKNN_SC = 1 #compute_SC(SFKNN_DPC1)
                KS_FDPC_SC = compute_SC(KS_FDPC1)

                # 个体公平性
                DPC_BPS = compute_boundary_point_similarity(DPC1, K=K)

                IFDPC_BPS = 1 #compute_boundary_point_similarity(IFDPC, K=K)
                IFDPC_BPS_improve = 1 - DPC_BPS / IFDPC_BPS

                SFKNN_BPS =1  # compute_boundary_point_similarity(SFKNN_DPC1, K=K)
                SFKNN_BPS_improve = 1 - DPC_BPS / SFKNN_BPS

                KS_FDPC_BPS = compute_boundary_point_similarity(KS_FDPC1, K=K)
                KS_FDPC_BPS_improve = 1 - DPC_BPS / KS_FDPC_BPS

                # 转换为三位小数
                IFDPC_BPS = round(IFDPC_BPS, 3)
                DPC_BPS = round(DPC_BPS, 3)
                SFKNN_BPS = round(SFKNN_BPS, 3)
                KS_FDPC_BPS = round(KS_FDPC_BPS, 3)

                IFDPC_BPS_improve = round(IFDPC_BPS_improve, 3)
                SFKNN_BPS_improve = round(SFKNN_BPS_improve, 3)
                KS_FDPC_BPS_improve = round(KS_FDPC_BPS_improve, 3)

                IFDPC_SC = round(IFDPC_SC, 3)
                DPC_SC = round(DPC_SC, 3)
                SFKNN_SC = round(SFKNN_BPS, 3)
                KS_FDPC_SC = round(KS_FDPC_SC, 3)

                results = {
                    '聚类数': cluster_num,
                    'K数': K,
                    'DPC方差比准则': DPC_SC,
                    'IFDPC方差比准则': IFDPC_SC,
                    'SFKNN-DPC方差比准则': SFKNN_SC,
                    'KS-FDPC方差比准则': KS_FDPC_SC,
                    'DPC个体公平性': DPC_BPS,
                    'IFDPC个体公平性': IFDPC_BPS,
                    'IFDPC个体公平性提升': IFDPC_BPS_improve,
                    'SFKNN-DPC个体公平性': SFKNN_BPS,
                    'SFKNN-DPC个体公平性提升': SFKNN_BPS_improve,
                    'KS-FDPC个体公平性': KS_FDPC_BPS,
                    'KS-FDPC个体公平性提升': KS_FDPC_BPS_improve
                }

                # Append results to the Excel file
                append_to_excel(file_name, results)
                print(f"Results for {data_name}, cluster={cluster_num}, K={K} appended.")
