# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

from IFDPC import get_varience
from IFDPC.plot_IFDPC import draw_cluster


class IFDPC3:
    def __init__(self, data, data_name, K=8, cluster_num=3):
        # 初始化数据集和参数
        self.data_name = data_name
        self.K = K
        self.dataset_cluster = data
        # 距离矩阵，相似矩阵
        self.dist_matrix, self.A = self.getDistanceMatrix()
        # 构建MKNN并构建互邻接矩阵
        self.MKNN, self.W = self.get_MKNN_adjacencyMatrix()
        # 计算密度，得到初始簇中心
        self.rho, self.deltas, initial_centers = self.get_rho_Icenters()
        # 子簇合并
        self.group_label = self.cluster_PD(cluster_num, initial_centers)

    # 计算距离矩阵和相似矩阵
    def getDistanceMatrix(self):
        # 使用欧氏距离计算距离矩阵
        dists = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        # 使用RBF核函数计算相似度矩阵
        gamma = 1.0 / self.dataset_cluster.shape[1]  # 经验值
        A = rbf_kernel(dists, gamma=gamma)
        return dists, A

    # 构建MKNN图
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



    def get_rho_Icenters(self):

        n = self.dataset_cluster.shape[0]
        rho = np.zeros(n)
        MKNN = self.MKNN

        # 局部密度
        for i in range(n):
            for j in MKNN[i]:
                if i != j:
                    rho[i] += 1 / (self.dist_matrix[i][j] + 1e-6)  # avoid division by zero
            rho[i] *= len(MKNN[i]) ** 2

        delta = np.zeros(n)
        # Calculate relative distance and identify initial cluster centers
        for i in range(n):
            _ = [self.dist_matrix[i][j] for j in range(n) if rho[j] > rho[i] and j in MKNN[i]]
            if len(_) > 0:
                delta[i] = np.min(_)

        initial_centers = np.where(delta == 0)[0]

        return rho, delta, initial_centers


    def cluster_PD(self, k, initial_centers):
        rho = self.rho
        dists = self.dist_matrix
        n = np.shape(rho)[0]
        label = -1 * np.ones(n).astype(int)
        for center in initial_centers:
            label[center] = center

        # 初始簇分配, 在互近邻矩阵中，与其他点都为0的，被当做了初始簇中心
        for i in range(n):
            if label[i] == -1:
                nearest_center = initial_centers[np.argmin(dists[i][initial_centers])]
                label[i] = nearest_center


        # unique_labels, counts = np.unique(label, return_counts=True)
        # # 找到簇中只有一个点的簇标签
        # singleton_clusters = unique_labels[counts == 1]
        # for cluster in singleton_clusters:
        #     # 获取该簇中的点的索引
        #     singleton_point = np.where(label == cluster)[0][0]
        #     min_dist = np.inf
        #     nearest_center = None
        #     # 找到最近的簇中心
        #     for center in initial_centers:
        #         if center != cluster:  # 排除自身簇
        #             if dists[singleton_point][center] < min_dist:
        #                 min_dist = dists[singleton_point][center]
        #                 nearest_center = center
        #     # 将该点分配给最近的簇
        #     label[singleton_point] = nearest_center
        #     if singleton_point in initial_centers:
        #         initial_centers = np.delete(initial_centers, np.where(initial_centers == singleton_point))

        unique_labels, counts = np.unique(label, return_counts=True)
        # 找到簇中只有一个点的簇标签
        singleton_clusters = unique_labels[counts == 1]
        print(singleton_clusters)


        # 直至聚类数
        # 直至聚类数 <= k
        # while np.unique(label).shape[0] > k:
        #     cur_cluster_ids = np.unique(label)
        #     # 计算每个簇的大小
        #     cluster_sizes = {cluster_id: np.sum(label == cluster_id) for cluster_id in cur_cluster_ids}
        #
        #     # 按照簇大小排序
        #     sorted_clusters = sorted(cur_cluster_ids, key=lambda cid: cluster_sizes[cid])
        #
        #     # 按排序的顺序计算合并力
        #     merged = False
        #     for i in range(len(sorted_clusters)):
        #         for j in range(i + 1, len(sorted_clusters)):
        #             if len(np.where(label == sorted_clusters[i])[0]) == 0 or len(
        #                     np.where(label == sorted_clusters[j])[0]) == 0:
        #                 break
        #             if i != j:
        #                 # 计算簇i和簇j的拉普拉斯矩阵特征值差异
        #                 delta_E = self.compute_merge_force(
        #                     np.where(label == sorted_clusters[i])[0],
        #                     np.where(label == sorted_clusters[j])[0],
        #                     label
        #                 )
        #
        #                 # 若合并有利于整体紧凑性，则合并
        #                 if delta_E < 0:
        #                     label = self.merge_clusters(sorted_clusters[i], sorted_clusters[j], label)
        #
        #                     break


        return label


    def compute_merge_force(self, cluster_i, cluster_j, label):
        # print(cluster_i)
        combined_points = np.concatenate([cluster_i, cluster_j])

        # 计算簇 i, j 以及合并后的簇的拉普拉斯矩阵并计算特征值
        L_i = self.get_laplacian(cluster_i)
        L_j = self.get_laplacian(cluster_j)
        L_ij = self.get_laplacian(combined_points)

        # 计算Fiedler值的变化
        lambda_i = self.get_fiedler_value(L_i)
        lambda_j = self.get_fiedler_value(L_j)
        lambda_ij = self.get_fiedler_value(L_ij)

        return lambda_ij - (lambda_i + lambda_j)

    def get_laplacian(self, cluster_points):
        """
        根据点的索引计算拉普拉斯矩阵
        :param cluster_points: 簇中点的索引
        :return: 簇的拉普拉斯矩阵
        """
        # print(f'cluster_points{cluster_points}')
        # print(f'np.ix_(cluster_points, cluster_points): {np.ix_(cluster_points, cluster_points)}')
        W = self.W[np.ix_(cluster_points, cluster_points)]  # 获取簇中点的邻接矩阵
        print(f'W{np.shape(W)}')
        D = np.diag(np.sum(W, axis=1))  # 度矩阵
        L = D - W  # 拉普拉斯矩阵
        return L


    def get_fiedler_value(self, L):
        # 计算第二小的特征值
        eigvals = np.linalg.eigvals(L)
        eigvals.sort()
        return eigvals[1]  # 第二小的特征值


    def merge_clusters(self, cluster_i, cluster_j, label):
        # 将所有属于 cluster_j 的点的标签更新为 cluster_i
        for idx in range(len(label)):
            if label[idx] == cluster_j:
                label[idx] = cluster_i
        return label

if __name__ == '__main__':
    all_datasets = ['hcvdat0',
                    # 'liver_disorder',
                    # 'iris'
                   # 'Wholesale', 'bank', 'Rice', 'obesity',
                    # 'drug_consumption', 'adult', 'abalone'
                   ]
    # 'Wholesale', 'bank', 'Rice', 'obesity', 'drug_consumption', 'adult'

    # dermatology数据都是nan, abalone含有nan
    for data_name in all_datasets:
        cluster_num = 5
        data = get_varience.read_data_IF(data_name)
        IFDPC = IFDPC3(data, data_name, K=4, cluster_num=cluster_num)
        print(data_name)
        print(f"IFDPC data_size:{np.shape(IFDPC.dataset_cluster)[0]}")

        print('-----------')
        #DPC1 = DPC(dataset, IFDPC.dataset_cluster, cluster_num=cluster_num)
        draw_cluster(IFDPC)

        #draw_cluster(DPC1)