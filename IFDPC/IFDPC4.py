# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

from IFDPC import get_varience
from IFDPC.DPC import DPC
from IFDPC.comment import *
from IFDPC.plot_IFDPC import *


class IFDPC4:
    def __init__(self, data, data_name, alpha, K=8, cluster_num=3):
        # 初始化数据集和参数
        self.data_name = data_name
        self.K = K
        self.dataset_cluster = data
        # 距离矩阵，相似矩阵
        self.dist_matrix, self.A = self.getDistanceMatrix()
        self.new_dist = self.get_newDis(alpha)
        self.dist_matrix = self.new_dist
        # 构建MKNN并构建互邻接矩阵
        self.MKNN, self.W = self.get_MKNN_adjacencyMatrix()
        # 计算密度，得到初始簇中心
        self.rho, self.deltas, self.initial_centers = self.get_rho_Icenters()
        # print(f'initial_centers: {initial_centers}')
        # 得到每个点的簇标签
        self.initial_label , self.initial_centers= self.initial_cluster()
        # 创建密度峰之间的互近邻矩阵，并合并密度峰之间的连通子图
        self.group_label = self.merge_clusters_with_dynamic_K(cluster_num)

    # 计算距离矩阵和相似矩阵
    def getDistanceMatrix(self):
        # 使用欧氏距离计算距离矩阵
        dists = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        print()
        # 使用RBF核函数计算相似度矩阵
        gamma = 1.0 / self.dataset_cluster.shape[1]  # 经验值
        A = rbf_kernel(dists, gamma=gamma)
        return dists, A


    def get_newDis(self, alpha=0.5):
        dists = self.dist_matrix
        # 将距离矩阵D映射到0-1区间
        D_min = np.min(dists)
        D_max = np.max(dists)
        normalized_dists = (dists - D_min) / (D_max - D_min)
        # 加权距离矩阵: D_weighted = D + alpha * A
        weighted_dists = normalized_dists + alpha * self.A

        return weighted_dists
    # 通过指定的K值构建全局的MKNN图
    def get_MKNN_adjacencyMatrix(self):

        distances = self.dist_matrix
        n = distances.shape[0]
        MKNN = [set() for _ in range(n)]

        nearest_neighbors = np.argsort(distances, axis=1)[:, 1:self.K+1]

        # 检查互为k近邻的点并构建MKNN集合
        for i in range(n):
            for j in nearest_neighbors[i]:
                if i in nearest_neighbors[j] and i != j:  # 如果i是j的k近邻，且j是i的k近邻
                    MKNN[i].add(j)
                    MKNN[j].add(i)  # 确保对称性

        # 互邻接矩阵
        W = np.zeros((n, n))
        for i in range(n):
            for j in MKNN[i]:
                W[i, j] = 1  # 互为近邻的点之间的边标记为1
                W[j, i] = 1
        return MKNN, W


    # 通过全局MKNN图和密度计算得到局部密度峰(初始聚类中心)
    def get_rho_Icenters(self):

        n = self.dataset_cluster.shape[0]
        rho = np.zeros(n)
        MKNN = self.MKNN

        # 局部密度
        for i in range(n):
            for j in MKNN[i]:
                if i != j:
                    rho[i] += 1 / (self.new_dist[i][j] + 1e-6)  # avoid division by zero
            rho[i] *= len(MKNN[i]) ** 2

        delta = np.zeros(n)
        # Calculate relative distance and identify initial cluster centers
        for i in range(n):
            _ = [self.new_dist[i][j] for j in range(n) if rho[j] > rho[i] and j in MKNN[i]]
            if len(_) > 0:
                delta[i] = np.min(_)

        initial_centers = np.where(delta == 0)[0]

        return rho, delta, initial_centers


    # 通过自适应的K值构建密度峰的MKNN图
    def get_peak_MKNN_matrix(self, initial_centers):
        n_data = self.dist_matrix.shape[0]

        # 获取密度峰之间的K近邻
        nearest_neighbors = [set() for _ in range(n_data)]
        for peak in initial_centers:
            KNN_neighbor_indices = np.argsort(
                self.new_dist[peak][initial_centers])  # Sort indices based on distances
            KNN_neighbor_peaks = [initial_centers[i] for i in
                                  KNN_neighbor_indices[1:self.K + 1]]  # Map sorted indices to initial_centers
            nearest_neighbors[peak] = KNN_neighbor_peaks

        MKNN = [set() for _ in range(n_data)]
        # 构建密度峰的互近邻矩阵
        for peak in initial_centers:
            for neighbor_peak in nearest_neighbors[peak]:
                if peak != neighbor_peak \
                        and neighbor_peak in nearest_neighbors[peak] \
                        and peak in nearest_neighbors[neighbor_peak]:
                    MKNN[peak].add(neighbor_peak)
                    MKNN[neighbor_peak].add(peak)

        # 得到
        peak_W_b = np.zeros((n_data, n_data))
        for peak in initial_centers:
            for neighbor_peak in MKNN[peak]:
                peak_W_b[peak, neighbor_peak] = 1  # Use the original indices for MKNN
                peak_W_b[neighbor_peak, peak] = 1  # Ensure symmetry
        return peak_W_b


    def find_connected_components(self, W_b, initial_centers):
        n = self.dataset_cluster.shape[0]
        visited = np.zeros(n, dtype=bool)
        clusters = []
        def dfs(peak, cluster):
            visited[peak] = True
            cluster.append(peak)
            for neighbor_peak in initial_centers:
                # print('----', neighbor_peak, W_b[peak, neighbor_peak])
                if W_b[peak, neighbor_peak] == 1 and not visited[neighbor_peak]:
                    dfs(neighbor_peak, cluster)

        # dfs寻找各个 连通子图（簇）
        for center in initial_centers:
            if not visited[center]:
                cluster = []
                dfs(center, cluster)
                clusters.append(cluster)

        return clusters

    def merge_clusters_with_dynamic_K(self, cluster_num, min_K=1, max_K=25):

        # while min_K < max_K:
        #     # 当前的 K 值是二分搜索的中点
        #     current_K = (min_K + max_K) // 2
        #     # print(current_K)
        #     self.K = current_K  # 更新 K 值
        #
        #     # 根据当前 K 值计算 MKNN 和互近邻值矩阵
        #     peak_W_b = self.get_peak_MKNN_matrix(initial_centers)
        #
        #     # 找到当前的连通子图（簇）
        #     clusters = self.find_connected_components(peak_W_b, initial_centers)
        #     num_clusters = len(clusters)  # 当前的簇数量
        #     # print(min_K, max_K, num_clusters, clusters)
        #     print(f"当前簇数量: {num_clusters}", f"min_K:{min_K}", f"max_K:{max_K}", f"current_K:{current_K}")
        #     # 如果簇数量大于期望值，增大 K 值；否则减小 K 值
        #     if num_clusters >= cluster_num:
        #         min_K = current_K + 1
        #     else :
        #         max_K = current_K
        for cur_K in range(max_K, min_K - 1, -1):  # 从 max_K 开始递减遍历
            self.K = cur_K  # 更新 K 值
            # print(f"当前 K 值: {cur_K}")

            # 根据当前 K 值计算 MKNN 和互近邻值矩阵
            peak_W_b = self.get_peak_MKNN_matrix(self.initial_centers)

            # 找到当前的连通子图（簇）
            clusters = self.find_connected_components(peak_W_b, self.initial_centers)
            num_clusters = len(clusters)  # 当前的簇数量

            # 打印当前簇数量和 K 值
            # print(f"当前簇数量: {num_clusters}, 当前 K 值: {cur_K}")

            # 如果簇数量 >= 期望值，退出循环
            if num_clusters >= cluster_num:
                break

        best_cluster_labels = self.initial_label.copy()
        initial_label = self.initial_label
        # 为每个连通子图的密度峰和对应的初始簇分配相同的标签
        for cluster_id, cluster in enumerate(clusters):
            for peak in cluster:
                for point, label in enumerate(initial_label):
                    if label == peak:
                        best_cluster_labels[point] = cluster_id
        # print(len(np.unique(best_cluster_labels) - 1))  # 除去簇标签为-1的点
        return best_cluster_labels




    def initial_cluster(self):
        rho = self.rho
        dists = self.new_dist
        n = np.shape(rho)[0]
        label = -1 * np.ones(n).astype(int)
        for center in self.initial_centers:
            label[center] = center

        # 初始簇分配, 在互近邻矩阵中，与其他点都为0的，被当做了初始簇中心
        for i in range(n):
            if label[i] == -1:
                nearest_center = self.initial_centers[np.argmin(dists[i][self.initial_centers])]
                label[i] = nearest_center

        unique_labels, counts = np.unique(label, return_counts=True)
        # 找到簇中只有一个点的簇标签
        singleton_clusters = unique_labels[counts == 1]
        label[singleton_clusters] = -1
        initial_centers = [center for center in self.initial_centers if center not in singleton_clusters]
        # print(singleton_clusters)
        return label, initial_centers

if __name__ == '__main__':
    all_datasets = ['hcvdat0',
                    'liver_disorder',
                    'iris',
                    # 'seeds',
                    'obesity',
                    'abalone',
                    # 'Room_Occupancy_Estimation',
                    'dabestic',
                    'athlete',
                   ]
    # 'Wholesale', 'bank', 'Rice', 'obesity', 'drug_consumption', 'adult'
    # dermatology数据都是nan, abalone含有nan
    for data_name in all_datasets:
        cluster_num = 5
        K = 10
        data = get_varience.read_data_IF(data_name)
        IFDPC = IFDPC4(data, data_name, alpha=0.01, K=K, cluster_num=cluster_num)
        print(data_name)
        print(f"IFDPC data_size:{np.shape(IFDPC.dataset_cluster)[0]}")

        print('-----------')
        DPC1 = DPC(data_name, IFDPC.dataset_cluster, cluster_num=cluster_num)
        draw_cluster(IFDPC)
        # draw_cluster(DPC1)

        #print('轮廓系数：--------')
        IFDPC_SC = compute_SC(IFDPC)
        print(f'IFDPC_SC: {IFDPC_SC}')
        DPC_SC = compute_SC(DPC1)
        print(f'DPC_SC: {DPC_SC}')

        #print('平均余弦相似度：--------')
        IFDPC_AC = compute_AC(IFDPC, k=K)
        print(f'IFDPC_AC: {IFDPC_AC}')
        DPC_AC = compute_AC(DPC1, k=K)
        print(f'DPC_AC: {DPC_AC}')
        # draw_graph(IFDPC)

        print()

