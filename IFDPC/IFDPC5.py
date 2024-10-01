# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

from FDPC.plot_FDPC import draw_cluster
from IFDPC.DPC_var import *
from IFDPC import get_varience
from IFDPC.DPC import DPC
from IFDPC.DPC_var.SFKNN_DPC__CCFB import SFKNN_DPC
from IFDPC.comment import *
from IFDPC.plot_IFDPC import *


class IFDPC5:
    def __init__(self, data, data_name, K, cluster_num):
        # 初始化数据集和参数
        self.data_name = data_name
        self.dataset_cluster = data
        # 距离矩阵，相似矩阵
        self.dist_matrix, self.A = self.getDistanceMatrix()

        self.n = np.shape(self.dataset_cluster)[0]
        self.initial_label = -1 * np.ones(self.n).astype(int)
        self.K = K
        # 构建KNN并构建互邻接矩阵
        self.KNN, self.MKNN = self.get_KNN_MKNN()
        # 计算密度，通过KNN和MKNN得到初始簇中心
        self.rho, self.peaks = self.get_rho_Icenters()
        self.pres = self.build_tree()
        self.group_label = self.initial_label
        # self.group_label = self.split_tree(self.pres, cluster_num - 1)
        # self.assign_remaining_points()

    # 计算距离矩阵和相似矩阵
    def getDistanceMatrix(self):

        dists = squareform(pdist(self.dataset_cluster, metric='euclidean'))

        # 去除距离为0的点对（距离为0表示两点相等）
        unique_indices = np.arange(self.dataset_cluster.shape[0])
        to_remove = set()

        for i in range(dists.shape[0]):
            for j in range(i + 1, dists.shape[1]):
                if dists[i, j] == 0:  # 如果距离为0，意味着两点相同
                    to_remove.add(j)  # 记录重复点的索引（只保留其中一个）

        # 重新组织数据集
        unique_indices = np.delete(unique_indices, list(to_remove))
        self.dataset_cluster = self.dataset_cluster[unique_indices]
        dists = squareform(pdist(self.dataset_cluster, metric='euclidean'))

        # 归一化距离矩阵
        # dists_min = np.min(dists)
        # dists_max = np.max(dists)
        # dists = (dists - dists_min) / (dists_max - dists_min)

        # 使用RBF核函数计算相似度矩阵
        gamma = 1.0 / self.dataset_cluster.shape[1]  # 经验值
        A = rbf_kernel(dists, gamma=gamma)
        # 归一化相似度矩阵 A
        A_min = np.min(A)
        A_max = np.max(A)
        A_normalized = (A - A_min) / (A_max - A_min)

        return dists, A_normalized


    def get_KNN_MKNN(self):

        KNN = np.argsort(self.dist_matrix, axis=1)[:, 1:self.K + 1]
        MKNN = [set() for _ in range(len(KNN))]

        # 检查互为k近邻的点并构建MKNN集合
        for i in range(self.n):
            for j in KNN[i]:
                if i in KNN[j] and i != j:
                    MKNN[i].add(j)
                    MKNN[j].add(i)  # 确保对称性
        return KNN, MKNN

    # 通过KNN, MKNN和密度计算得到局部密度峰(初始聚类中心)
    def get_rho_Icenters(self):
        dists = self.dist_matrix
        n = self.dataset_cluster.shape[0]
        rho = np.zeros(n)
        KNN = self.KNN

        # 局部密度
        for i in range(n):
            for j in KNN[i]:
                if i != j:
                    rho[i] += np.exp(-(dists[i][j] + 1e-6))

        peaks = []
        for i in range(n):
            knn_rho_values = rho[KNN[i]]  # rho 是每个点的密度
            center = KNN[i][np.argmax(knn_rho_values)]
            peaks.append(center)
            # 在密度峰的KNN集合P_k中找到平均相似度最高的点q
            # P_k = KNN[center]
            # avg_similarities = np.array([
            #     np.mean(self.A[neighbor, P_k][P_k != neighbor])
            #     for neighbor in P_k if neighbor != center
            # ])
            # max_avg_index = np.argmax(avg_similarities)
            # q = P_k[max_avg_index]
            # # 选择初始中心
            # if center == q:
            #     peaks.append(center)
            # else:
            #     # 在KNN(center)中找到点a，使得 S(a, q) + S(a, p) 最大
            #     knn_of_center = KNN[center]
            #     similarities = self.A[knn_of_center[:, None], [q, center]]  # 获取S(a, q)和S(a, p)
            #     similarity_sums = similarities.sum(axis=1)  # 计算S(a, q) + S(a, p)
            #     # 找到最大相似度和对应的点a
            #     best_a_index = np.argmax(similarity_sums)
            #     best_a = knn_of_center[best_a_index] if best_a_index < len(knn_of_center) else None
            #     peaks.append(best_a)
#
        peaks = list(set(peaks))
        # 初始簇，只标记中心
        for center in peaks:
            self.initial_label[center] = center  # 只有非离群点被标记为簇中心

        return rho, peaks

    def build_tree(self):
        peaks = self.peaks
        rho = self.rho
        n = len(rho)

        pres = np.full(n, -1, dtype=int)  # 初始化每个点的父节点
        for center in peaks:
            # 找到所有密度比当前峰大的邻居
            neighbors = [c for c in peaks if c != center and rho[c] > rho[center]]
            if len(neighbors) > 0:
                # 找到密度更大且最相似的峰
                closest_neighbor_index = np.argmax(self.A[center, neighbors])
                fa = neighbors[closest_neighbor_index]
                pres[center] = fa  # 设置父节点

        return pres

    def split_tree(self, pres, k):
        """从 pres 中删除相似度最低的 k 条边，然后通过DFS遍历连通分量划分簇"""

        n = len(pres)
        similarities = []
        peaks = self.peaks
        peaks_edges = [(p, pres[p]) for p in peaks if pres[p] != -1]  # 只考虑peaks之间的边

        # 遍历每个峰及其父节点，计算它们的相似度
        for point, parent in peaks_edges:
            similarity = self.A[point, parent]  # 相似度
            similarities.append((similarity, point, parent))

        # 按相似度降序排列，选择相似度最低的 k 条边作为割边
        similarities.sort(reverse=False, key=lambda x: x[0])
        cut_edges = similarities[:k]  # 选择相似度最低的 k 条边

        # 从 pres 中删除相似度最低的 k 条边
        for _, point, parent in cut_edges:
            # print(point, parent)
            pres[point] = -1  # 将这个点的父节点设为 -1，表示删除边

        # 初始化每个节点的簇编号
        clusters = np.full(n, -1, dtype=int)
        cluster_id = 0

        # 对所有节点进行 DFS 遍历，找到每个连通分量
        for peak in peaks:
            if clusters[peak] == -1:  # 如果当前点未被分配到任何簇
                # 对该点进行 DFS，找到所有与该点连通的节点，并分配给同一个簇
                self.dfs(peak, pres, clusters, cluster_id)
                cluster_id += 1
        return clusters

    def dfs(self, point, pres, clusters, cluster_id):
        """递归地进行DFS，将连通的节点分配到同一个簇"""
        # 将当前节点分配到当前簇
        clusters[point] = cluster_id

        # 遍历每个节点，检查是否是该节点的子节点
        for child, parent in enumerate(pres):
            if parent == point and clusters[child] == -1:  # 如果当前点是父节点，且子节点未被访问
                # 递归访问子节点
                self.dfs(child, pres, clusters, cluster_id)

        # 还要检查当前节点是否有父节点，且父节点未被访问（因为是有向图）
        if pres[point] != -1 and clusters[pres[point]] == -1:
            self.dfs(pres[point], pres, clusters, cluster_id)

    def find_nearest_in_cluster(self, point, clusters):
        cluster_points = np.where(clusters == clusters[point])[0]  # 找到与当前点同簇的点
        # 排除当前点本身
        cluster_points = cluster_points[cluster_points != point]
        # 如果簇中只有当前点，直接返回 None
        if len(cluster_points) == 0:
            return None
        # 找到簇内最近点的索引（与当前点相似度最大）
        nearest_point = cluster_points[np.argmax(self.A[point, cluster_points])]
        return nearest_point

    def assign_remaining_points(self):
        # 初始化隶属度矩阵（memberships），初始时未分配的点隶属度为0
        unassigned_points = np.where(self.group_label == -1)[0]
        memberships = np.zeros((len(unassigned_points), len(self.peaks)))
        # 初始化未分配点到各簇的隶属度，使用与密度峰之间的相似度
        for idx, point_idx in enumerate(unassigned_points):
            for c_idx, peak_idx in enumerate(self.peaks):
                memberships[idx, c_idx] = 1 / (1 + self.dist_matrix[point_idx, peak_idx])  # 基于距离的隶属度

        # Step 2: 按照算法流程进行迭代
        while np.max(memberships) > 0:
            # 找到具有最大隶属度的点及其对应的簇
            s = np.unravel_index(np.argmax(memberships, axis=None), memberships.shape)[0]  # 点索引
            c = np.argmax(memberships[s])  # 对应的簇索引
            point_idx = unassigned_points[s]  # 点的全局索引
            # 确定该点属于该簇
            self.group_label[point_idx] = c
            # 清除这个点的隶属度，避免重复选择
            memberships[s, :] = 0
            # 更新该点的K近邻点的隶属度
            neighbors = self.KNN[point_idx]
            for neighbor_idx in neighbors:
                if self.group_label[neighbor_idx] == -1:  # 仅更新未分配的点
                    for c_idx in range(len(self.peaks)):
                        memberships[unassigned_points == neighbor_idx, c_idx] += self.calculate_fuzzy_membership(
                            neighbor_idx, c_idx, self.KNN, self.group_label)
        return self.group_label

    def calculate_fuzzy_membership(self, i, c, knn_indices, group_label):

        neighbors = knn_indices[i]  # 点 i 的 K 近邻
        gamma = self.A[i, neighbors]  # 计算 gamma 权重

        # 计算隶属度，累计近邻中已分配到簇 c 的点的隶属度
        p_c_i = np.sum(gamma[group_label[neighbors] == c])
        return p_c_i

if __name__ == '__main__':
    all_datasets = [
        #  'Wholesale',
        # 'obesity',
        # 'abalone',
        'seeds',# 可以
        # 'iris'
        # 'hcvdat0',
        # 'athlete',
                    # 'bank', # 可以用
                    # 'census1990',  # 可以用
                    # 'creditcard', # 可以用
                    # 'adult'
                   ]

    # 'Wholesale', 'bank', 'Rice', 'obesity', 'drug_consumption'(拉了), ‘athlete(拉了)’, 'adult'(个体公平提升小),'Rice'(提升小), 'Room_Occupancy_Estimation'(提升小), # 可以,  'Wholesale', # 可以用
    # 'iris','obesity', 'hcvdat0', # 可以        # 'Rice', 'dabestic', 'liver_disorder', 'glass', 'wdbc', 'wine'
    for data_name in all_datasets:
        print(data_name)
        cluster_num = 5
        K = 7
        data = get_varience.read_data_IF(data_name)
        print(f'size:{len(data)}')
        IFDPC = IFDPC5(data, data_name, K, cluster_num=cluster_num)
        print('-----------')
        # DPC1 = DPC(data_name, IFDPC.dataset_cluster, cluster_num=cluster_num)
        #
        # SFKNN_DPC1 = SFKNN_DPC(data, IFDPC.K, cluster_num)
        # SFKNN_DPC1.data_name = data_name
        # SFKNN_DPC1.fit()

        draw_cluster_IFDPC(IFDPC)
        # draw_cluster_DPC(DPC1)
        # draw_cluster(SFKNN_DPC1)
        # #
        # print('CH指数：--------')
        # IFDPC_SC = compute_SC(IFDPC)
        # SFKNN_SC = compute_SC(SFKNN_DPC1)
        # DPC_SC = compute_SC(DPC1)
        # print(f'DPC_CH: {DPC_SC}')
        # print(f'IFDPC_CH: {IFDPC_SC}')
        # print(f'SFKNN_CH: {SFKNN_SC}')
        # # # #
        # print('边界点的个体公平性：--------')
        # # 边界点个体公平性
        # DPC_BPS = compute_boundary_point_similarity(DPC1, K=IFDPC.K)
        # print(f'DPC_BPS: {DPC_BPS}')
        # IFDPC_BPS = compute_boundary_point_similarity(IFDPC, K=IFDPC.K)
        # print(f'IFDPC_BPS: {IFDPC_BPS}')
        # SFKNN_DPC_BPS = compute_boundary_point_similarity(SFKNN_DPC1, K=IFDPC.K)
        # print(f'SFKNN_DPC_BPS: {SFKNN_DPC_BPS}')
        print()

# 局部密度峰的第二种选取方式
# _ = [dists[i][j] for j in range(n) if rho[j] > rho[i] and j in self.KNN[i]]
# if len(_) == 0:
#     center = i
#     # 在密度峰的KNN集合P_k中找到平均相似度最高的点q
#     P_k = KNN[center]
#     avg_similarities = np.array([
#         np.mean(self.A[neighbor, P_k][P_k != neighbor])
#         for neighbor in P_k if neighbor != center
#     ])
#     max_avg_index = np.argmax(avg_similarities)
#     q = P_k[max_avg_index]
#     # 选择初始中心
#     if center == q:
#         peaks.append(center)
#     else:
#         # 在KNN(center)中找到点a，使得 S(a, q) + S(a, p) 最大
#         knn_of_center = KNN[center]
#         similarities = self.A[knn_of_center[:, None], [q, center]]  # 获取S(a, q)和S(a, p)
#         similarity_sums = similarities.sum(axis=1)  # 计算S(a, q) + S(a, p)
#         # 找到最大相似度和对应的点a
#         best_a_index = np.argmax(similarity_sums)
#         best_a = knn_of_center[best_a_index] if best_a_index < len(knn_of_center) else None
#         peaks.append(best_a)
# peaks.append(i) # center = i