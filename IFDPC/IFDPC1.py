# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from DPC import DPC
import get_varience

from IFDPC.plot_IFDPC import draw_cluster
from scipy.linalg import eigh



# 对照组
class IFDPC1:
    def __init__(self, data_name, cluster_num):

        # 读数据库名字得到数据集，单属性所有敏感属性组，数据集各敏感属性组的占比
        self.data_name = data_name
        self.dataset_cluster = get_varience.read_data_IF(data_name)
        self.dist_matrix = self.getDistanceMatrix()
        self.laplacian_matrix = self.get_walk_laplacian_matrix(k=15, gamma=1)
        self.rho_matrix = self.get_local_density(method="Laplacian")
        self.deltas, self.nearest_neiber = self.get_deltas()
        self.group_centers, self.group_label  = self.cluster_PD(cluster_num)

        # 计算距离矩阵

    def getDistanceMatrix(self):
        return squareform(pdist(self.dataset_cluster, metric='euclidean'))

    # 计算游走拉普拉斯矩阵
    def get_walk_laplacian_matrix(self, k, gamma):
        # 构建邻接矩阵 A
        N = np.shape(self.dist_matrix)[0]
        # 经验值
        if gamma is None:
            gamma = 1.0 / self.dist_matrix.shape[1]
        # 基于数据分布的设置
        # gamma = 1.0 / (2 * np.mean(dist_matrix))
        # 相似矩阵
        A = rbf_kernel(self.dist_matrix, gamma=gamma)
        self.A = A
        # 度矩阵 D
        D = np.diag(np.sum(A, axis=1))
        # 游走拉普拉斯矩阵 L_rw
        print(D)
        L_rw = np.eye(N) - np.linalg.inv(D).dot(A)

        # 对拉普拉斯矩阵进行谱分解，提取前 k 个特征向量作为密度估计基础
        eigenvals, eigenvecs = eigh(L_rw)
        return eigenvecs[:, :min(k, N)]  # 使用前 k 个特征向量


    # 使用游走拉普拉斯矩阵计算局部密度
    def get_local_density(self, method=None):
        N = np.shape(self.dataset_cluster)[0]
        rho = np.zeros(N)
        dists = self.dist_matrix
        if method == "Laplacian":
            # 基于游走拉普拉斯特征向量的密度计算
            laplacian = self.laplacian_matrix
            for i in range(N):
                # 使用特征向量的平方和作为局部密度
                rho[i] = np.sum(laplacian[i, :] ** 2)

        # else:
        #     for i in range(N):
        #         if method == None:  # 截断核
        #             rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        #         else:
        #             rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
        return rho
# knn
        # 密度较大点相对距离

    def get_deltas(self):
        dists = self.dist_matrix
        rho = self.rho_matrix

        N = np.shape(dists)[0]
        deltas = np.zeros(np.shape(dists)[0])
        nearest_neiber = np.zeros(N)
        rho_sorted_point_id = np.argsort(-rho)
        for i, point in enumerate(rho_sorted_point_id):
            if i == 0:
                continue
            point_higher_rho = rho_sorted_point_id[:i]
            deltas[point] = np.min(dists[point, point_higher_rho])
            point_mindist_to = np.argmin(dists[point, point_higher_rho])
            nearest_neiber[point] = point_higher_rho[point_mindist_to].astype(int)

        deltas[rho_sorted_point_id[0]] = np.max(deltas)
        return deltas, nearest_neiber


    def cluster_PD(self, k):
        rho = self.rho_matrix  # 每个点的密度
        similarity_matrix = self.A  # 相似度矩阵
        N = np.shape(rho)[0]
        label = -1 * np.ones(N).astype(int)
        nearest_neiber = self.nearest_neiber

        # 初始化中心点集合
        centers = []

        # 找到密度最高的点作为第一个中心点
        first_center = np.argmax(rho)
        centers.append(first_center)
        label[first_center] = 0  # 第一个中心点标签为0

        # 查找其他中心点
        while len(centers) < k:
            # 对每个尚未标记为中心的点，计算其与当前所有中心的相似度加权密度
            unlabelled = [i for i in range(N) if label[i] == -1]
            max_density_index = -1
            max_value = -np.inf

            for i in unlabelled:
                # 计算与现有中心点的相似度加权密度
                similarity_scores = (1 -  similarity_matrix[i, centers])
                total_similarity = np.sum(similarity_scores)

                # 选择最高的加权密度作为新的中心点
                if total_similarity * rho[i] > max_value:
                    max_value = total_similarity * rho[i]
                    max_density_index = i

            # 更新中心点和标签
            centers.append(max_density_index)
            label[max_density_index] = len(centers) - 1

        # 将密度从大到小排序
        index_rho = np.argsort(-rho)
        for i, point in enumerate(index_rho):

            # 从密度大的点进行标号
            if label[point] == -1:
                # 如果没有被标记过
                # 那么聚类标号与距离其最近且密度比其大的点的标号相同
                # 密度比当前点大的，一定已经先排好位置了
                label[point] = label[int(nearest_neiber[point])]
        return centers, label



if __name__ == '__main__':
    all_dataset = ['hcvdat0', 'liver_disorder',
                   'Wholesale', 'bank', 'Rice', 'obesity', 'drug_consumption', 'adult']
    #  'Wholesale','bank', 'hcvdat0', 'Rice', 'obesity', 'drug_consumption','adult'
    # all_dataset = ['dermatology']
    # dermatology数据都是nan
    for dataset in all_dataset:
        cluster_num = 5
        IFDPC = IFDPC1(dataset, cluster_num=cluster_num)
        DPC1 = DPC(dataset, IFDPC.dataset_cluster, cluster_num=cluster_num)
        draw_cluster(IFDPC)
        draw_cluster(DPC1)