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
        # "dpc_balance": np.array([
        #     [0.1269, 0.3083, 0.3082, 0.3082, 0.3082],
        #     [0.2505, 0.2466, 0.2535, 0.2535, 0.2535],
        #     [0.2144, 0.2112, 0.2112, 0.2112, 0.2112],
        #     [0.1838, 0.1810, 0.1811, 0.1592, 0.1592],
        #     [0.2575, 0.1392, 0.1393, 0.2395, 0.2395]
        # ]),
        "fdpc_balance": np.array([
            [0.4890, 0.5280, 0.5820, 0.5660, 0.5490],
            [0.4820, 0.5270, 0.5710, 0.5530, 0.5270],
            [0.4800, 0.5130, 0.5700, 0.5520, 0.4860],
            [0.4720, 0.5100, 0.5560, 0.5350, 0.4770],
            [0.4690, 0.5000, 0.5470, 0.5200, 0.4840]
        ]),
        "ks_fdpc_balance": np.array([
            [0.4060, 0.3810, 0.3820, 0.3810, 0.4010],
            [0.4060, 0.3810, 0.3820, 0.3810, 0.4010],
            [0.4060, 0.3810, 0.3820, 0.3810, 0.4010],
            [0.4060, 0.3810, 0.3820, 0.3810, 0.4010],
            [0.4060, 0.3810, 0.3820, 0.3810, 0.4010]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.5400, 0.5010, 0.4810, 0.5300, 0.3780],
            [0.4900, 0.4900, 0.4760, 0.5250, 0.3720],
            [0.4880, 0.4870, 0.4870, 0.5550, 0.3900],
            [0.4770, 0.5050, 0.4890, 0.5540, 0.4060],
            [0.4570, 0.4950, 0.4870, 0.5520, 0.4040]
        ])
    },
    "Athlete": {
        # "dpc_balance": np.array([
        #     [0.8232, 0.8018, 0.7923, 0.7962, 0.7475],
        #     [0.8410, 0.7626, 0.7323, 0.7278, 0.7290],
        #     [0.8160, 0.7172, 0.7569, 0.7110, 0.7386],
        #     [0.7872, 0.7429, 0.7369, 0.7344, 0.7228],
        #     [0.7762, 0.7277, 0.7329, 0.7403, 0.7361]
        # ]),
        "fdpc_balance": np.array([
            [0.6060, 0.6360, 0.6910, 0.6830, 0.7060],
            [0.6060, 0.6360, 0.6910, 0.6830, 0.7060],
            [0.6060, 0.6360, 0.6910, 0.6830, 0.7060],
            [0.6060, 0.6360, 0.6910, 0.6830, 0.7060],
            [0.6020, 0.6330, 0.6900, 0.6810, 0.7050]
        ]),
        "ks_fdpc_balance": np.array([
            [0.3990, 0.3840, 0.3760, 0.3570, 0.3410],
            [0.3990, 0.3840, 0.3760, 0.3570, 0.3410],
            [0.3990, 0.3840, 0.3760, 0.3570, 0.3410],
            [0.3990, 0.3840, 0.3760, 0.3570, 0.3410],
            [0.3990, 0.3840, 0.3760, 0.3570, 0.3410]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.5280, 0.5090, 0.3190, 0.3260, 0.4360],
            [0.4330, 0.3710, 0.2730, 0.2280, 0.2070],
            [0.2500, 0.2850, 0.1960, 0.2730, 0.1380],
            [0.2730, 0.2410, 0.1570, 0.1970, 0.1470],
            [0.3560, 0.1940, 0.1360, 0.1210, 0.1280]
        ])
    },
    "Bank": {
        # "dpc_balance": np.array([
        #     [0.6439, 0.6389, 0.4397, 0.4406, 0.4406],
        #     [0.6474, 0.5036, 0.4954, 0.4972, 0.5064],
        #     [0.5846, 0.4751, 0.4671, 0.4686, 0.4763],
        #     [0.5745, 0.4908, 0.4733, 0.5112, 0.5112],
        #     [0.5699, 0.4938, 0.5063, 0.5041, 0.4476]
        # ]),
        "fdpc_balance": np.array([
            [0.6120, 0.6450, 0.6660, 0.6200, 0.5750],
            [0.5930, 0.6200, 0.6120, 0.5430, 0.4840],
            [0.5910, 0.6140, 0.5930, 0.5230, 0.4470],
            [0.5860, 0.6080, 0.5570, 0.5190, 0.4390],
            [0.5730, 0.5920, 0.5500, 0.4960, 0.4290]
        ]),
        "ks_fdpc_balance": np.array([
            [0.4080, 0.3860, 0.3800, 0.3600, 0.3630],
            [0.4080, 0.3860, 0.3800, 0.3600, 0.3630],
            [0.4080, 0.3860, 0.3800, 0.3600, 0.3630],
            [0.4080, 0.3860, 0.3800, 0.3600, 0.3630],
            [0.4080, 0.3860, 0.3800, 0.3600, 0.3630]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.4580, 0.3750, 0.4630, 0.4070, 0.3890],
            [0.3960, 0.3900, 0.4720, 0.4420, 0.3710],
            [0.3980, 0.4450, 0.4670, 0.4600, 0.3960],
            [0.4220, 0.4610, 0.4640, 0.4420, 0.3840],
            [0.4190, 0.4490, 0.4620, 0.4340, 0.4010]
        ])
    },
    "Obesity": {
        # "dpc_balance": np.array([
        #     [0.4893, 0.4508, 0.5423, 0.5980, 0.5843],
        #     [0.5458, 0.3059, 0.3287, 0.5007, 0.4898],
        #     [0.3568, 0.3819, 0.3031, 0.4323, 0.4232],
        #     [0.3343, 0.3426, 0.2598, 0.4264, 0.3572],
        #     [0.3447, 0.3666, 0.3115, 0.3686, 0.3552]
        # ]),
        "fdpc_balance": np.array([
            [0.5820, 0.6440, 0.6390, 0.6300, 0.6440],
            [0.5820, 0.6420, 0.6160, 0.6300, 0.6390],
            [0.5810, 0.6310, 0.5950, 0.5970, 0.6120],
            [0.5760, 0.6260, 0.5910, 0.5950, 0.5800],
            [0.5750, 0.6240, 0.5810, 0.5880, 0.5560]
        ]),
        "ks_fdpc_balance": np.array([
            [0.4000, 0.3790, 0.3950, 0.3710, 0.3440],
            [0.4000, 0.3790, 0.3950, 0.3710, 0.3440],
            [0.4000, 0.3790, 0.3950, 0.3710, 0.3440],
            [0.4000, 0.3790, 0.3950, 0.3710, 0.3440],
            [0.4000, 0.3790, 0.3950, 0.3710, 0.3440]
        ]),
        "sfknn_dpc_balance": np.array([
            [0.3720, 0.3610, 0.4930, 0.5240, 0.5950],
            [0.4520, 0.5500, 0.6770, 0.5190, 0.4240],
            [0.4030, 0.5500, 0.4310, 0.3610, 0.2350],
            [0.4430, 0.3820, 0.3660, 0.3870, 0.2570],
            [0.4330, 0.3910, 0.4880, 0.3330, 0.2210]
        ])
    },
    # "Seeds": {
    #     # "dpc_balance": np.array([
    #     #     [0.0744, 0.3402, 0.3423, 0.0735, 0.1695],
    #     #     [0.0595, 0.2543, 0.2482, 0.0588, 0.1286],
    #     #     [0.0496, 0.2119, 0.2006, 0.0490, 0.0869],
    #     #     [0.0425, 0.1792, 0.1720, 0.0420, 0.0745],
    #     #     [0.0372, 0.1568, 0.1505, 0.0368, 0.0652]
    #     # ]),
    #     "fdpc_balance": np.array([
    #         [0.6100, 0.6730, 0.6570, 0.6720, 0.5700],
    #         [0.5710, 0.6440, 0.6480, 0.6610, 0.5660],
    #         [0.5450, 0.6480, 0.6480, 0.6280, 0.5060],
    #         [0.5520, 0.6160, 0.6240, 0.6280, 0.4820],
    #         [0.5400, 0.5720, 0.6230, 0.5530, 0.4440]
    #     ]),
    #     "ks_fdpc_balance": np.array([
    #         [0.3900, 0.5450, 0.5450, 0.5150, 0.5140],
    #         [0.4180, 0.5020, 0.5210, 0.4470, 0.4960],
    #         [0.4750, 0.4770, 0.5210, 0.4780, 0.4960],
    #         [0.4890, 0.4770, 0.5210, 0.4780, 0.4960],
    #         [0.4890, 0.4770, 0.5210, 0.4780, 0.4960]
    #     ]),
    #     "sfknn_dpc_balance": np.array([
    #         [0.4770, 0.6820, 0.7310, 0.7370, 0.7330],
    #         [0.4550, 0.6620, 0.7110, 0.7190, 0.7230],
    #         [0.4480, 0.6300, 0.7550, 0.7640, 0.7700],
    #         [0.4290, 0.5670, 0.7350, 0.7400, 0.7490],
    #         [0.4890, 0.6190, 0.6460, 0.7220, 0.7700]
    #     ])
    # },
    "Wholesale": {
        # "dpc_balance": np.array([
        #     [0.6940, 0.7491, 0.5639, 0.5615, 0.3998],
        #     [0.6573, 0.7026, 0.6017, 0.4506, 0.3199],
        #     [0.6282, 0.5878, 0.5026, 0.5009, 0.2671],
        #     [0.5378, 0.5727, 0.4056, 0.4293, 0.3242],
        #     [0.4778, 0.4791, 0.4198, 0.3537, 0.3692]
        # ]),
        "fdpc_balance": np.array([
            [0.4660, 0.4990, 0.5540, 0.5960, 0.6050],
            [0.4650, 0.4960, 0.5450, 0.5640, 0.5960],
            [0.4640, 0.4930, 0.5400, 0.5580, 0.5880],
            [0.4620, 0.4790, 0.5260, 0.5610, 0.5570],
            [0.4570, 0.4760, 0.5260, 0.5390, 0.4980]
        ]),
        "ks_fdpc_balance": np.array([
            [0.2850, 0.2670, 0.2400, 0.2450, 0.2370],
            [0.2850, 0.2670, 0.2400, 0.2450, 0.2370],
            [0.2850, 0.2670, 0.2400, 0.2450, 0.2370],
            [0.2850, 0.2670, 0.2400, 0.2450, 0.2370],
            [0.2850, 0.2670, 0.2400, 0.2450, 0.2370]

        ]),
        "sfknn_dpc_balance": np.array([
            [0.4290, 0.3470, 0.3480, 0.3990, 0.3170],
            [0.4140, 0.3150, 0.3670, 0.4240, 0.3630],
            [0.3990, 0.3000, 0.4150, 0.4750, 0.4380],
            [0.5120, 0.4190, 0.5060, 0.4740, 0.4460],
            [0.4430, 0.3580, 0.4570, 0.4490, 0.4300]
        ])
    }
}
# Normalize colors using the global min and max across all datasets
global_min = min(
                 [np.min(data["fdpc_balance"]) for data in datasets.values()] +
                 [np.min(data["ks_fdpc_balance"]) for data in datasets.values()] +
                 [np.min(data["sfknn_dpc_balance"]) for data in datasets.values()])
global_max = max(
                 [np.max(data["fdpc_balance"]) for data in datasets.values()] +
                 [np.max(data["ks_fdpc_balance"]) for data in datasets.values()] +
                 [np.max(data["sfknn_dpc_balance"]) for data in datasets.values()])

fig = plt.figure(figsize=(20, 30))

cluster_counts = np.array([4, 5, 6, 7, 8])
dc_ratios = np.array([6, 7, 8, 9, 10])
K_s = np.arange(6, 10)

plot_idx = 1
for dataset_name, data in datasets.items():
    # FDPC Bar Plot
    ax = fig.add_subplot(len(datasets), 3, plot_idx, projection='3d')
    X, Y = np.meshgrid(cluster_counts, dc_ratios)
    X, Y = X.flatten(), Y.flatten()
    Z = data["fdpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'IFDPC SPC in {dataset_name}', 'Cluster Count', 'K', 'SPC', plt.cm.viridis, global_min, global_max, 0.4, 0.4)
    plot_idx += 1

    # KS-FDPC Bar Plot
    ax = fig.add_subplot(len(datasets), 3, plot_idx, projection='3d')
    X, Y = np.meshgrid(cluster_counts, dc_ratios)
    X, Y = X.flatten(), Y.flatten()
    Z = data["ks_fdpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'KS-FDPC SPC in {dataset_name}', 'Cluster Count', 'K', 'SPC', plt.cm.viridis, global_min, global_max, 0.4, 0.4)
    plot_idx += 1

    # SFKNN-DPC Bar Plot
    ax = fig.add_subplot(len(datasets), 3, plot_idx, projection='3d')
    Z = data["sfknn_dpc_balance"].T.flatten()
    plot_balance(ax, X, Y, Z, f'SFKNN-DPC SPC in {dataset_name}', 'Cluster Count', 'K', 'SPC', plt.cm.viridis, global_min, global_max, 0.4, 0.4)
    plot_idx += 1

    # DPC Bar Plot
    # ax = fig.add_subplot(len(datasets), 4, plot_idx, projection='3d')
    # X, Y = np.meshgrid(cluster_counts, dc_ratios)
    # X, Y = X.flatten(), Y.flatten()
    # Z = data["dpc_balance"].T.flatten()
    # plot_balance(ax, X, Y, Z, f'DPC Balance in {dataset_name}', 'Cluster Count', 'dc Ratio', 'Balance', plt.cm.viridis, global_min, global_max, 0.4, 0.004)
    # plot_idx += 1

plt.tight_layout(pad=5.0, h_pad=5.0, w_pad=2.0)
plt.savefig('all_datasets_balance_plots.png')
plt.show()