# -*- coding: utf-8 -*-
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def draw_cluster_IFDPC(dpc):
    datas = dpc.dataset_cluster
    df = pd.DataFrame(datas)
    dimension = len(df.columns) - 1

    # 设置高DPI，提升图像分辨率
    plt.figure(figsize=(10, 8), dpi=500)  # 设置DPI和图像大小

    label = dpc.group_label
    unique_labels = np.unique(label)

    # 去除黑色，确保黑色专门用于噪声点
    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan",
                       "magenta", "#88c999"])

    # 如果维度大于2，使用PCA进行降维
    if dimension > 2:
        datas = PCA(n_components=2).fit_transform(datas)

    for i, label_id in enumerate(unique_labels):
        if label_id == -1:  # 标记噪声点，使用黑色
            plt.scatter(datas[label == label_id, 0], datas[label == label_id, 1],
                        c='black', s=10, label='Noise')  # 噪声点颜色为黑色
        else:
            color = colors[i % len(colors)]  # 选择非黑色的颜色
            plt.scatter(datas[label == label_id, 0], datas[label == label_id, 1],
                        c=color, s=40, label=f'Cluster {label_id}')  # 簇的颜色

    # 绘制 LSCs 之间的 pres 路径
    pres = dpc.pres
    # 遍历所有 LSCs，绘制父节点和子节点之间的路径
    for i, parent in enumerate(pres):
        if parent != -1:  # 如果 pres[i] 有父节点
            # 获取父节点和子节点的坐标
            child_coords = datas[i]
            parent_coords = datas[parent]

            # 绘制一条从子节点到父节点的线
            plt.plot([child_coords[0], parent_coords[0]], [child_coords[1], parent_coords[1]],
                     'k--', linewidth=2)  # 虚线连接子节点和父节点
    # 绘制图表并添加标题和图例
    plt.title(dpc.data_name)
    # 如果提供了保存路径，则保存图片
    plt.savefig( dpc.data_name + '1', bbox_inches='tight')  # 保存图片，去除多余的边缘
    plt.show()


def draw_cluster_DPC(dpc):
    datas = dpc.dataset_cluster
    centers = dpc.group_centers
    # 将 NumPy 数组转换为 DataFrame 对象
    df = pd.DataFrame(datas)
    # 问 DataFrame 对象的 columns 属性
    dimension = len(df.columns) - 1

    plt.figure(figsize=(10, 8), dpi=500)

    label = dpc.group_label
    unique_labels = np.unique(label)
    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan",
                       "magenta", "beige", "hotpink", "#88c999", "black"])

    if dimension > 2:
        datas = PCA(n_components=2).fit_transform(datas)  # 如果属性数量大于2，降维

    for i, label_id in enumerate(unique_labels):
        if label_id == -1:  # 噪声点
            plt.scatter(datas[label == label_id, 0], datas[label == label_id, 1], c='black', s=7, label='Noise')
        else:
            color = colors[i % len(colors)]
            plt.scatter(datas[label == label_id, 0], datas[label == label_id, 1], c=color, s=7,
                        label=f'Cluster {label_id}')
    # 特别处理聚类中心，使用'+'标记
    plt.scatter(datas[centers, 0], datas[centers, 1], color='k', marker='+', s=200, label='Cluster Centers')
    # 添加标题
    plt.title(dpc.data_name)
    plt.legend()
    plt.show()

def draw_graph(IFDPC):
    datas = IFDPC.dataset_cluster
    n = datas.shape[0]
    # 将相似度矩阵的上三角部分拉直成一个向量
    similarity_values = IFDPC.A[np.triu_indices(n, k=1)]

    # 绘制相似度分布的直方图
    plt.figure(figsize=(8, 6))
    plt.hist(similarity_values, bins=30, density=True, alpha=0.7, color='g')
    plt.title('Distribution of Pairwise Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# 绘制决策图
def draw_decision(fdpc):
    rho = fdpc.rho
    deltas = fdpc.deltas
    data_matrix = fdpc.dataset_cluster

    plt.cla()
    for i in range(np.shape(data_matrix)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    # plt.savefig(filename+"_decision.jpg")
    plt.show()