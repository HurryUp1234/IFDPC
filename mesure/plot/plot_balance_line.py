# -*- coding: utf-8 -*-
# Sample data for plotting
import pandas as pd
import matplotlib.pyplot as plt

# Sample data creation (Assuming this data is derived from previous computations)
data_fdpc = {
   "Cluster Count": [4, 5, 6, 7, 8],
   "Balance": [0.51434, 0.60684, 0.63784, 0.64544, 0.53576]  # 计算均值结果
}

data_dpc = {
   "Cluster Count": [4, 5, 6, 7, 8],
   "Balance": [0.31904, 0.29796, 0.2408, 0.22134, 0.2125]  # 计算均值结果
}

data_ks_fdpc = {
   "Cluster Count": [4, 5, 6, 7, 8],
   "Balance": [0.27875, 0.27995, 0.301475, 0.287075, 0.30995]  # 计算均值结果
}

data_sfknn_dpc = {
   "Cluster Count": [4, 5, 6, 7, 8],
   "Balance": [0.288725, 0.31165, 0.25345, 0.235625, 0.21155]  # 计算均值结果
}

# Convert to DataFrame
df_fdpc = pd.DataFrame(data_fdpc).set_index("Cluster Count")
df_dpc = pd.DataFrame(data_dpc).set_index("Cluster Count")
df_ks_fdpc = pd.DataFrame(data_ks_fdpc).set_index("Cluster Count")
df_sfknn_dpc = pd.DataFrame(data_sfknn_dpc).set_index("Cluster Count")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_fdpc.index, df_fdpc["Balance"], marker='o', linestyle='-', linewidth=2, label='FDPC')
plt.plot(df_dpc.index, df_dpc["Balance"], marker='s', linestyle='--', linewidth=2, label='DPC')
plt.plot(df_ks_fdpc.index, df_ks_fdpc["Balance"], marker='^', linestyle='-.', linewidth=2, label='KS-FDPC')
plt.plot(df_sfknn_dpc.index, df_sfknn_dpc["Balance"], marker='d', linestyle=':', linewidth=2, label='SFKNN-DPC')

plt.xlabel('Cluster Count', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.title('Rice', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

