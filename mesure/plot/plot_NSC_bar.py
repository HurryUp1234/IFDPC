# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 表1: DPC 和 FDPC 数据
data_dpc_fdpc = {
    "Cluster Count": [4, 5, 6, 7, 8],
    "FDPC_SC": [0.51792, 0.5331, 0.5351, 0.530045, 0.52984],
    "DPC_SC": [0.54531, 0.51234, 0.49141, 0.49388, 0.47408]
}
data_ks_fdpc_sfknn_dpc = {
    "Cluster Count": [4, 5, 6, 7, 8],
    "KS-FDPC_SC": [0.51385, 0.51485, 0.508625, 0.5152125, 0.5099],
    "SFKNN-DPC_SC": [0.44455, 0.4443025, 0.4448125, 0.446, 0.44625]
}


df_dpc_fdpc = pd.DataFrame(data_dpc_fdpc)
df_ks_fdpc_sfknn_dpc = pd.DataFrame(data_ks_fdpc_sfknn_dpc)

# 计算均值
fdpc_mean_balance = df_dpc_fdpc.groupby("Cluster Count")["FDPC_SC"].mean()
dpc_mean_balance = df_dpc_fdpc.groupby("Cluster Count")["DPC_SC"].mean()
ks_fdpc_mean_balance = df_ks_fdpc_sfknn_dpc.groupby("Cluster Count")["KS-FDPC_SC"].mean()
sfknn_dpc_mean_balance = df_ks_fdpc_sfknn_dpc.groupby("Cluster Count")["SFKNN-DPC_SC"].mean()

# 数据准备
cluster_counts = fdpc_mean_balance.index
bar_width = 0.2

# 绘制二维条形图
plt.figure(figsize=(12, 6))
r1 = np.arange(len(cluster_counts))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

plt.bar(r1, fdpc_mean_balance.values, color='b', width=bar_width, edgecolor='grey', label='FDPC')
plt.bar(r2, dpc_mean_balance.values, color='g', width=bar_width, edgecolor='grey', label='DPC')
plt.bar(r3, ks_fdpc_mean_balance.values, color='r', width=bar_width, edgecolor='grey', label='KS-FDPC')
plt.bar(r4, sfknn_dpc_mean_balance.values, color='c', width=bar_width, edgecolor='grey', label='SFKNN-DPC')

plt.xlabel('Cluster Count', fontsize=14)
plt.ylabel('NSC Value', fontsize=14)
plt.title('Adult', fontsize=16)
plt.xticks([r + bar_width for r in range(len(cluster_counts))], cluster_counts, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.ylim(0, 1)  # 设置 y 轴范围为 0 到 1
plt.show()