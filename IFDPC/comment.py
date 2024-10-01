import numpy as np
from pyparsing import White
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.metrics import normalized_mutual_info_score

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict, Counter

from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel


def compute_SC(fdpc):
    # 计算所有样本的轮廓系数
    # silhouette_avg = silhouette_score(fdpc.dataset_cluster, fdpc.group_label)
    db_score = davies_bouldin_score(fdpc.dataset_cluster, fdpc.group_label)
    # ch_score = calinski_harabasz_score(fdpc.dataset_cluster, fdpc.group_label)
    return db_score




def get_NMI(fdpc, dpc):
    # group_label_true 和 group_label_pred 分别是原始和改进后的聚类结果标签
    return normalized_mutual_info_score(fdpc.group_label, dpc.group_label)



# 计算边界点相似召回度
def compute_boundary_point_similarity(model, K):
    clusters = model.group_label
    n = len(clusters)
    total_similarity = 0.0  # 用于存储所有边界点的个体相似度总和
    boundary_point_count = 0  # 用于统计边界点的个数

    dists = model.dist_matrix
    # 使用RBF核函数计算相似度矩阵
    gamma = 1.0 / model.dataset_cluster.shape[1]  # 经验值
    A = rbf_kernel(dists, gamma=gamma)
    # 归一化相似度矩阵 A 到 0-1 之间
    A_min = np.min(A)
    A_max = np.max(A)
    A_normalized = (A - A_min) / (A_max - A_min)

    # 将对角线上的自身相似度设为负无穷大，以确保不会被选中
    np.fill_diagonal(A_normalized, -np.inf)

    # 对每行进行排序，选择相似度最大的 K 个点
    KNN = np.argsort(A_normalized, axis=1)[:, -K:]  # 找到每行最大的 K 个点的索引
    # 对每个点进行统计
    for i in range(n):
        same_cluster_count = 1
        knn_neighbors = KNN[i]  # 获取点 i 的 K 近邻
        is_boundary = False  # 用于标记该点是否是边界点
        cluster_diff_num = 0
        # 统计 K 近邻中与点 i 属于同一簇的点数量
        for neighbor in knn_neighbors:
            if clusters[neighbor] == clusters[i]:  # 如果属于同一个簇
                same_cluster_count += 1
            else:
                cluster_diff_num += 1
                is_boundary = True  # 存在不属于同一簇的邻居，标记为边界点

        if is_boundary:  # 只对边界点计算相似度
            boundary_point_count += 1
            # 计算该点的同簇比例
            similarity = same_cluster_count / len(knn_neighbors) * 1.0 /cluster_diff_num
            total_similarity += similarity

    # 计算边界点的平均相似度，如果没有边界点，返回 0
    if boundary_point_count == 0:
        return 0.0
    else:
        average_similarity = total_similarity / boundary_point_count
        return average_similarity