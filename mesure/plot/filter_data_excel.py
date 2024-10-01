# -*- coding: utf-8 -*-

import pandas as pd


def modify_excel(input_file, output_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 过滤掉聚类数为3并且dc为0.01, 0.03, 0.05, 0.07, 0.09的行
    df_filtered = df[~((df['聚类数量'] == 3) | (df['dc比例'].isin([0.01, 0.03, 0.05, 0.07, 0.09])))]

    # 删除与AED和AWD相关的列
    columns_to_drop = ['FDPC_AED', 'DPC_AED', 'FDPC_AWD', 'DPC_AWD', 'AED下降率', 'AWD下降率']
    df_filtered = df_filtered.drop(columns=columns_to_drop)

    # 新表名
    sheet_name = 'modified'

    # 写入新的Excel文件
    with pd.ExcelWriter(output_file) as writer:
        df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)


# 示例调用
project_path = 'E:\\work\\fair_cluster\\current_work\\fdpc\\'
all_datasets = ['liver_disorder', 'Wholesale',
                'bank', 'hcvdat0', 'Rice', 'vertebral', 'adult', 'athlete']

for data_name in all_datasets:
    input_file = project_path + data_name + 'excel.xlsx'
    output_file = project_path + data_name + 'modified.xlsx'
    print(f"Processing {data_name}...")
    modify_excel(input_file, output_file)
    print(f"{data_name} processed and saved as {output_file}")