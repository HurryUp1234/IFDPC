# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# Function to plot balance
def plot_balance(ax, X, Y, Z, title, xlabel, ylabel, zlabel, colors, global_min, global_max, dx, dy):
    colors_normalized = plt.cm.viridis((Z - global_min) / (global_max - global_min))
    ax.bar3d(X, Y, np.zeros(len(Z)), dx, dy, Z, color=colors_normalized, zsort='average')
    # 使用 ax.text2D 在图表上方添加标题
    ax.text2D(0.5, 0.93, title, transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
    # 设置 x 轴标签
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.xaxis.labelpad = 10  # 调整 x 轴标签距离

    # 设置 y 轴标签
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.yaxis.labelpad = 10  # 调整 y 轴标签距离

    # 设置 z 轴标签
    ax.set_zlabel(zlabel, fontsize=10, fontweight='bold')
    ax.zaxis.labelpad = 10  # 调整 z 轴标签距离

    # 设置 z 轴范围
    ax.set_zlim(0, 1)



# Data definitions for each dataset
datasets = {
    "Abalone": {
        "dpc_balance": np.array([
            [0.1269, 0.3083, 0.3082, 0.3082, 0.3082],
            [0.2505, 0.2466, 0.2535, 0.2535, 0.2535],
            [0.2144, 0.2112, 0.2112, 0.2112, 0.2112],
            [0.1838, 0.1810, 0.1811, 0.1592, 0.1592],
            [0.2575, 0.1392, 0.1393, 0.2395, 0.2395]
        ]),
        "fdpc_balance": np.array([
            [0.8016, 0.7596, 0.8149, 0.8120, 0.7996],
            [0.7592, 0.6038, 0.6330, 0.6439, 0.6110],
            [0.9310, 0.8104, 0.9135, 0.8496, 0.6471],
            [0.7972, 0.7537, 0.7372, 0.5699, 0.6264],
            [0.9051, 0.7962, 0.6165, 0.4928, 0.3555]
        ]),
        "ks_fdpc_balance": np.array([
            [0.3829, 0.4513, 0.2435, 0.3111],
            [0.4456, 0.4408, 0.2948, 0.2488],
            [0.3946, 0.3914, 0.2738, 0.3190],
            [0.4168, 0.4026, 0.2715, 0.3009],
            [0.3667, 0.3522, 0.2376, 0.2662]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.3351, 0.1736, 0.1202, 0.1181],
            [0.2681, 0.1995, 0.2044, 0.2100],
            [0.2234, 0.1665, 0.1707, 0.1753],
            [0.3158, 0.1427, 0.1463, 0.1503],
            [0.2768, 0.1282, 0.1297, 0.1315]
        ])
    },
    "Adult": {
        "dpc_balance": np.array([
            [0.8232, 0.8018, 0.7923, 0.7962, 0.7475],
            [0.8410, 0.7626, 0.7323, 0.7278, 0.7290],
            [0.8160, 0.7172, 0.7569, 0.7110, 0.7386],
            [0.7872, 0.7429, 0.7369, 0.7344, 0.7228],
            [0.7762, 0.7277, 0.7329, 0.7403, 0.7361]
        ]),
        "fdpc_balance": np.array([
            [0.9630, 0.9577, 0.9642, 0.9164, 0.9496],
            [0.9673, 0.9781, 0.9587, 0.9145, 0.8878],
            [0.9504, 0.9785, 0.9163, 0.8906, 0.9172],
            [0.9760, 0.9521, 0.9432, 0.9463, 0.9047],
            [0.9630, 0.9106, 0.9754, 0.9069, 0.9045]
        ]),
        "ks_fdpc_balance": np.array([
            [0.8169, 0.8559, 0.9377, 0.8828],
            [0.8174, 0.8779, 0.8779, 0.7871],
            [0.8411, 0.8190, 0.8859, 0.7640],
            [0.8185, 0.7884, 0.8511, 0.7918],
            [0.8293, 0.8044, 0.8556, 0.7810]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.2496, 0.2497, 0.2494, 0.5822],
            [0.3898, 0.1999, 0.1998, 0.3636],
            [0.3785, 0.1663, 0.3249, 0.3881],
            [0.1428, 0.1428, 0.1427, 0.2157],
            [0.1250, 0.1249, 0.1247, 0.1886]
        ])
    },
    "Hcvdat0": {
        "dpc_balance": np.array([
            [0.6439, 0.6389, 0.4397, 0.4406, 0.4406],
            [0.6474, 0.5036, 0.4954, 0.4972, 0.5064],
            [0.5846, 0.4751, 0.4671, 0.4686, 0.4763],
            [0.5745, 0.4908, 0.4733, 0.5112, 0.5112],
            [0.5699, 0.4938, 0.5063, 0.5041, 0.4476]
        ]),
        "fdpc_balance": np.array([
            [0.8755, 0.8407, 0.9456, 0.9863, 0.9620],
            [0.8549, 0.8574, 0.8122, 0.7797, 0.9446],
            [0.8963, 0.9028, 0.9020, 0.9550, 0.7780],
            [0.9961, 0.8862, 0.9832, 0.8178, 0.8526],
            [0.9246, 0.9942, 0.9791, 0.7688, 0.8179]
        ]),
        "ks_fdpc_balance": np.array([
            [0.6111, 0.6437, 0.7690, 0.6066],
            [0.6032, 0.6319, 0.6777, 0.5877],
            [0.6304, 0.6226, 0.5787, 0.5832],
            [0.6141, 0.6063, 0.5644, 0.5561],
            [0.6252, 0.6036, 0.5561, 0.5270]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.6568, 0.6839, 0.5708, 0.4101],
            [0.6229, 0.5465, 0.5483, 0.3286],
            [0.5279, 0.4662, 0.5984, 0.2736],
            [0.4525, 0.3998, 0.5261, 0.2342],
            [0.4567, 0.4418, 0.4608, 0.2048]
        ])
    },
    "Obesity": {
        "dpc_balance": np.array([
            [0.4893, 0.4508, 0.5423, 0.5980, 0.5843],
            [0.5458, 0.3059, 0.3287, 0.5007, 0.4898],
            [0.3568, 0.3819, 0.3031, 0.4323, 0.4232],
            [0.3343, 0.3426, 0.2598, 0.4264, 0.3572],
            [0.3447, 0.3666, 0.3115, 0.3686, 0.3552]
        ]),
        "fdpc_balance": np.array([
            [0.7407, 0.9513, 0.6877, 0.7791, 0.9430],
            [0.7980, 0.7446, 0.8064, 0.8323, 0.7808],
            [0.8325, 0.9659, 0.9258, 0.9013, 0.6819],
            [0.8545, 0.9252, 0.7443, 0.6670, 0.4435],
            [0.8528, 0.8030, 0.7285, 0.6054, 0.3847]
        ]),
        "ks_fdpc_balance": np.array([
            [0.8407, 0.6335, 0.6873, 0.8980],
            [0.4668, 0.6664, 0.6221, 0.8271],
            [0.4884, 0.6795, 0.6448, 0.8472],
            [0.5420, 0.5797, 0.5736, 0.7718],
            [0.4993, 0.5854, 0.5201, 0.6492]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.3544, 0.4857, 0.5692, 0.3873],
            [0.1987, 0.4322, 0.4568, 0.4095],
            [0.2377, 0.1652, 0.3797, 0.3419],
            [0.2066, 0.1948, 0.4632, 0.4336],
            [0.2816, 0.2948, 0.3893, 0.2749]
        ])
    },
    "Room": {
        "dpc_balance": np.array([
            [0.0744, 0.3402, 0.3423, 0.0735, 0.1695],
            [0.0595, 0.2543, 0.2482, 0.0588, 0.1286],
            [0.0496, 0.2119, 0.2006, 0.0490, 0.0869],
            [0.0425, 0.1792, 0.1720, 0.0420, 0.0745],
            [0.0372, 0.1568, 0.1505, 0.0368, 0.0652]
        ]),
        "fdpc_balance": np.array([
            [0.7484, 0.7331, 0.7452, 0.7456, 0.7494],
            [0.7960, 0.7926, 0.7953, 0.7957, 0.7691],
            [0.8112, 0.8177, 0.8272, 0.8296, 0.8225],
            [0.8419, 0.8358, 0.8508, 0.8457, 0.8355],
            [0.8669, 0.8684, 0.8549, 0.8660, 0.8533]
        ]),
        "ks_fdpc_balance": np.array([
            [0.1703, 0.1705, 0.1844, 0.2038],
            [0.1362, 0.1364, 0.1266, 0.1362],
            [0.1135, 0.1019, 0.0660, 0.1092],
            [0.0674, 0.0874, 0.0512, 0.0663],
            [0.0475, 0.0764, 0.0448, 0.0489]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.2481, 0.2479, 0.2485, 0.2496],
            [0.1992, 0.1991, 0.1983, 0.1996],
            [0.1651, 0.1649, 0.1653, 0.1655],
            [0.1417, 0.1416, 0.1419, 0.1422],
            [0.1238, 0.1237, 0.1238, 0.1237]
        ])
    },
    "Wholesale": {
        "dpc_balance": np.array([
            [0.6940, 0.7491, 0.5639, 0.5615, 0.3998],
            [0.6573, 0.7026, 0.6017, 0.4506, 0.3199],
            [0.6282, 0.5878, 0.5026, 0.5009, 0.2671],
            [0.5378, 0.5727, 0.4056, 0.4293, 0.3242],
            [0.4778, 0.4791, 0.4198, 0.3537, 0.3692]
        ]),
        "fdpc_balance": np.array([
            [0.7345, 0.9159, 0.9345, 0.9706, 0.9818],
            [0.9340, 0.9067, 0.8921, 0.7771, 0.9039],
            [0.8148, 0.9202, 0.7871, 0.8688, 0.8904],
            [0.9351, 0.8288, 0.9012, 0.8226, 0.8120],
            [0.8477, 0.8308, 0.8309, 0.8238, 0.8199]
        ]),
        "ks_fdpc_balance": np.array([
            [0.8253, 0.8148, 0.6788, 0.7585],
            [0.7673, 0.7399, 0.5244, 0.6646],
            [0.6117, 0.7344, 0.3504, 0.4052],
            [0.6304, 0.6861, 0.2989, 0.3485],
            [0.5731, 0.5909, 0.2603, 0.3054]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.6158, 0.4025, 0.4025, 0.5130],
            [0.4963, 0.4435, 0.4087, 0.3249],
            [0.4177, 0.2756, 0.2698, 0.2713],
            [0.3562, 0.2357, 0.2339, 0.2334],
            [0.3128, 0.2047, 0.2032, 0.2049]
        ])
    }
}
# Normalize colors using the global min and max across all datasets
global_min = min([np.min(data["dpc_balance"]) for data in datasets.values()] +
                 [np.min(data["fdpc_balance"]) for data in datasets.values()] +
                 [np.min(data["ks_fdpc_balance"]) for data in datasets.values()] +
                 [np.min(data["sfknn_dpc_balance"]) for data in datasets.values()])
global_max = max([np.max(data["dpc_balance"]) for data in datasets.values()] +
                 [np.max(data["fdpc_balance"]) for data in datasets.values()] +
                 [np.max(data["ks_fdpc_balance"]) for data in datasets.values()] +
                 [np.max(data["sfknn_dpc_balance"]) for data in datasets.values()])

fig = plt.figure(figsize=(20, 30))

cluster_counts = np.array([4, 5, 6, 7, 8])
dc_ratios = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
K_s = np.arange(5, 9)

plot_idx = 1
for dataset_name, data in datasets.items():
    # FDPC Bar Plot
    ax = fig.add_subplot(len(datasets), 4, plot_idx, projection='3d')
    X, Y = np.meshgrid(cluster_counts, dc_ratios)
    X, Y = X.flatten(), Y.flatten()
    Z = data["fdpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'FDPC Balance in {dataset_name}', 'Cluster Count', 'dc Ratio', 'Balance', plt.cm.viridis, global_min, global_max, 0.4, 0.004)
    plot_idx += 1

    # KS-FDPC Bar Plot
    ax = fig.add_subplot(len(datasets), 4, plot_idx, projection='3d')
    X, Y = np.meshgrid(cluster_counts, K_s)
    X, Y = X.flatten(), Y.flatten()
    Z = data["ks_fdpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'KS-FDPC Balance in {dataset_name}', 'Cluster Count', 'K', 'Balance', plt.cm.viridis, global_min, global_max, 0.4, 0.4)
    plot_idx += 1

    # SFKNN-DPC Bar Plot
    ax = fig.add_subplot(len(datasets), 4, plot_idx, projection='3d')
    Z = data["sfknn_dpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'SFKNN-DPC Balance in {dataset_name}', 'Cluster Count', 'K', 'Balance', plt.cm.viridis, global_min, global_max, 0.4, 0.4)
    plot_idx += 1

    # DPC Bar Plot
    ax = fig.add_subplot(len(datasets), 4, plot_idx, projection='3d')
    X, Y = np.meshgrid(cluster_counts, dc_ratios)
    X, Y = X.flatten(), Y.flatten()
    Z = data["dpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'DPC Balance in {dataset_name}', 'Cluster Count', 'dc Ratio', 'Balance', plt.cm.viridis, global_min, global_max, 0.4, 0.004)
    plot_idx += 1

plt.tight_layout(pad=5.0, h_pad=5.0, w_pad=2.0)
plt.savefig('all_datasets_balance_plots.png')
plt.show()