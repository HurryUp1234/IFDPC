# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from DPC import DPC
import get_varience

from plot_IFDPC import draw_cluster
from scipy.linalg import eigh



# 对照组
class IFDPC2:
    def __init__(self, data_name, cluster_num, k=8):

        # 读数据库名字得到数据集，单属性所有敏感属性组，数据集各敏感属性组的占比
        self.K = k
        self.data_name = data_name
        self.dataset_cluster = get_varience.read_data_IF(data_name)
        # 距离矩阵，相似度矩阵
        self.dist_matrix, self.A = self.getDistanceMatrix()
        self.MKNN = self.get_MKNN(k)
        self.rho, self.deltas, initial_centers = self.get_LocalDensity_RDis()
        # 子簇合并
        self.group_label = self.cluster_PD(cluster_num, initial_centers)

    # 距离矩阵，相似矩阵
    def getDistanceMatrix(self):
        dists = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        # 经验值
        gamma = 1.0 / self.dataset_cluster.shape[1]
        A = rbf_kernel(dists, gamma=gamma)
        return dists, A


    def get_MKNN(self, k=8):
        distances = self.dist_matrix
        n = distances.shape[0]
        MKNN = [set() for _ in range(n)]
        # MKNN[i] : 点i的MKNN集合
        nearest_neighbors = np.argsort(distances, axis=1)[:, 1:k+1]

        # Check for mutual nearest neighbors
        for i in range(n):
            for j in nearest_neighbors[i]:
                # Check if i is in j's k-nearest neighbors
                if i in nearest_neighbors[j]:
                    MKNN[i].add(j)
        return MKNN

    #
    def get_LocalDensity_RDis(self):
        n = self.dataset_cluster.shape[0]

        rho = np.zeros(n)
        MKNN = self.MKNN

        # Calculate local density
        for i in range(n):
            for j in MKNN[i]:
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
        rho = self.rho  # 每个点的密度
        dists = self.dist_matrix
        # A = self.A  # 相似度矩阵
        n = np.shape(rho)[0]
        label = -1 * np.ones(n).astype(int)
        for center in initial_centers:
            label[center] = center

        # 得到初始子簇
        # 为每个非中心点分配簇
        for i in range(n):
            if label[i] == -1:  # 如果点i未被分配
                # 寻找距离最近的中心点
                min_dist = np.inf
                nearest_center = None
                for center in initial_centers:
                    if self.dist_matrix[i][center] < min_dist:
                        min_dist = self.dist_matrix[i][center]
                        nearest_center = center
                label[i] = nearest_center
        # 直至聚类数 < K
        while (np.unique(label).shape[0] > self.K):
            # 计算每个簇i的K个连接数最多的其他簇集合
            KMN = self.compute_connections(label)
            # 判断，然后得到中心簇
            center_clusters = self.get_center_clusters(label, KMN)
            # 其余子簇通过计算与中心簇平均相似度，与相似度最高的中心簇进行合并
            label = self.merge_clusters(label, center_clusters)
        # 返回最终簇
        return label

    # 子簇合并
    def merge_clusters(self, labels, center_clusters):

        # 计算簇间平均相似度
        CA = self.compute_CA(labels)

        # 合并非中心簇到相似度最高的中心簇
        new_labels = labels.copy()
        for cluster_id in labels:
            if cluster_id not in center_clusters:
                print(CA[cluster_id][center_clusters])
                similarities = {center: CA[cluster_id, center] for center in center_clusters}
                print(similarities)
                best_match = max(similarities, key=similarities.get)
                new_labels[labels == cluster_id] = best_match

        return new_labels

    def get_center_clusters(self, labels, KMN):
        unique_clusters = np.unique(labels)
        MKMN = {}
        AD = self.compute_AD(labels)  # 计算各个簇的平均密度

        # 计算每个簇的MKMN集合
        for cluster_id in unique_clusters:
            MKMN[cluster_id] = [other_cluster for other_cluster in KMN[cluster_id] if cluster_id in KMN[other_cluster]]

        # 确定密度中心簇
        center_clusters = []
        for cluster_id in unique_clusters:
            if not MKMN[cluster_id]:
                continue

            # 从AD字典中获取所有相关簇的密度
            densities = [AD[other_cluster] for other_cluster in MKMN[cluster_id]]

            # 找出密度最大的簇索引
            max_density_index = np.argmax(densities)

            # 检查这个最大密度是否属于当前簇
            if MKMN[cluster_id][max_density_index] == cluster_id:
                center_clusters.append(cluster_id)

        return center_clusters

    #子簇间的平均相似度
    def compute_CA(self, labels):
        unique_clusters = np.unique(labels)
        CA = np.zeros((len(unique_clusters), len(unique_clusters)))
        for i, ci in enumerate(unique_clusters):
            for j, cj in enumerate(unique_clusters):
                if ci != cj:
                    indices_i = np.where(labels == ci)[0]
                    indices_j = np.where(labels == cj)[0]
                    similarities = [self.A[p, q] for p in indices_i for q in indices_j]
                    CA[i, j] = np.mean(similarities)
        return CA

    # 子簇间的平均密度
    def compute_AD(self, labels):
        # 计算每个子簇的平均密度
        unique_labels = np.unique(labels)
        AD = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            AD[label] = np.mean(self.rho[indices])

        return AD

    def compute_SD(self, labels):
        # 计算每个子簇的平均密度
        unique_labels = np.unique(labels)
        AP = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            AP[label] = np.mean(self.rho[indices])

        # 计算子簇间的平均密度相似度
        unique_labels = list(AP.keys())
        n = len(unique_labels)
        Pavg = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Pavg_i = AP[unique_labels[i]]
                    Pavg_j = AP[unique_labels[j]]
                    Pavg[i, j] = 2 * np.sqrt(Pavg_i * Pavg_j) / (Pavg_i + Pavg_j + 1e-6) # 有0的情况
                else:
                    Pavg[i, j] = 0  # 自身与自身的相似度设置为0，或者可以选择为1根据需要调整
        gamma = 1.0 / Pavg.shape[1]
        SD = rbf_kernel(Pavg, gamma=gamma)
        return SD

    def compute_CON(self, labels):
        # 计算子簇间的集中度
        unique_labels = np.unique(labels)
        CON = np.zeros((len(unique_labels), len(unique_labels)))
        for i, ci in enumerate(unique_labels):
            for j, cj in enumerate(unique_labels):
                if i != j:
                    indices_i = np.where(labels == ci)[0]
                    indices_j = np.where(labels == cj)[0]
                    sum_dists = 0
                    count = 0
                    for a in indices_i:
                        for b in indices_j:
                            if b in self.MKNN[a]:
                                sum_dists += self.dist_matrix[a, b]
                                count += 1
                    CON[i, j] = 1 / sum_dists * count if count > 0 else np.inf  # 防止除以零
        return CON

    # 得到每个子簇的KMN
    def compute_connections(self, labels):
        unique_clusters = np.unique(labels)
        KMN = {}

        # 计算每个簇之间的MKNN连接数
        connection_counts = {}

        # 初始化每个簇之间的连接计数
        for cluster_id in unique_clusters:
            connection_counts[cluster_id] = {other_cluster: 0 for other_cluster in unique_clusters if
                                             other_cluster != cluster_id}

        # 遍历每个簇，计算符合MKNN条件的连接数
        for ci in unique_clusters:
            indices_ci = np.where(labels == ci)[0]
            for cj in unique_clusters:
                if cj != ci:
                    indices_cj = np.where(labels == cj)[0]
                    for i in indices_ci:
                        for j in indices_cj:
                            # 检查是否i和j互为MKNN
                            if j in self.MKNN[i] and i in self.MKNN[j]:
                                connection_counts[ci][cj] += 1

        # 为每个簇找出K个连接数最多的其他簇
        for cluster_id in unique_clusters:
            sorted_connections = sorted(connection_counts[cluster_id].items(), key=lambda x: x[1], reverse=True)
            KMN[cluster_id] = [other_cluster for other_cluster, _ in sorted_connections[:self.K]]

        return KMN


if __name__ == '__main__':
    all_dataset = ['hcvdat0', 'liver_disorder', 'iris'
                   # 'Wholesale', 'bank', 'Rice', 'obesity',
                    # 'drug_consumption', 'adult', 'abalone'
                   ]
    # 'Wholesale', 'bank', 'Rice', 'obesity', 'drug_consumption', 'adult'

    # dermatology数据都是nan, abalone含有nan
    for dataset in all_dataset:
        cluster_num = 5
        IFDPC = IFDPC2(dataset, cluster_num=cluster_num)
        print(dataset)
        print(f"IFDPC data_size:{np.shape(IFDPC.dataset_cluster)[0]}")

        print('-----------')
        #DPC1 = DPC(dataset, IFDPC.dataset_cluster, cluster_num=cluster_num)
        draw_cluster(IFDPC)
        #draw_cluster(DPC1)