import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import get_varience

# 对照组
class DPC:
    def __init__(self, data_name,  dataset_cluster, cluster_num=None):

        # 读数据库名字得到数据集，单属性所有敏感属性组，数据集各敏感属性组的占比
        self.data_name = data_name
        self.dataset_cluster = dataset_cluster
        # 距离矩阵
        self.dist_matrix = squareform(pdist(self.dataset_cluster, metric='euclidean'))
        self.dc = self.select_dc()

        # 各点局部密度(原密度),默认用截断核, 可用 Gaussion
        self.rho = self.get_local_density(method="Gaussion")

        # print('-----------------------')
        # 各点的中心偏移距离
        self.deltas, self.nearest_neiber = self.get_deltas()
        # 选取聚类中心
        self.group_centers = self.find_centers_auto(cluster_num)
        # 聚类分组
        self.group_label = self.cluster_PD()

    def getDistanceMatrix(self):
        return squareform(pdist(self.dataset_cluster, metric='euclidean'))

    # 选出dc
    def select_dc(self):
        dists = self.dist_matrix
        N = np.shape(dists)[0]
        tt = np.reshape(dists, N * N)  # 转一维方便排序
        percent = 2.0
        position = int(N * (N - 1) * percent / 100)
        dc = np.sort(tt)[position + N]
        return dc

    # 局部密度
    def get_local_density(self, method=None):  # 方法：高斯核，但默认为截断核

        dists = self.dist_matrix
        dc = self.dc
        N = np.shape(dists)[0]
        rho = np.zeros(N)

        for i in range(N):
            if method == None:  # 截断核
                rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
            else:
                rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
        return rho

    # 密度较大点相对距离
    def get_deltas(self):
        dists = self.dist_matrix
        rho = self.rho

        N = np.shape(dists)[0]
        deltas = np.zeros(np.shape(dists)[0])
        nearest_neiber = np.zeros(N)
        # 将密度从大到小排序
        rho_sorted_point_id = np.argsort(-rho)
        for i, point in enumerate(rho_sorted_point_id):  # i:表示大小顺序  point ：表示元素，此处代表点的编号
            # 对于密度最大的点
            if i == 0:
                continue

            point_higher_rho = rho_sorted_point_id[:i]
            # 存距离
            deltas[point] = np.min(dists[point, point_higher_rho])

            # 存点编号(可以删)
            point_mindist_to = np.argmin(dists[point, point_higher_rho])
            nearest_neiber[point] = point_higher_rho[point_mindist_to].astype(int)

        deltas[rho_sorted_point_id[0]] = np.max(deltas)
        return deltas, nearest_neiber

    def find_centers_auto(self, k=None):
        rho = self.rho
        deltas = self.deltas
        centers = []

        if k == 0 or k == None:
            rho_threshold = (np.min(rho) + np.max(rho)) / 2
            delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
            N = np.shape(rho)[0]

            for i in range(N):
                if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
                    centers.append(i)
            return np.array(centers)

        else:
            rho_delta = self.rho * self.deltas
            centers = np.argsort(-rho_delta)
            centers = centers[:k]

        return centers

    # 基于相对距离的正向选点（原方法）
    def cluster_PD(self):
        centers = self.group_centers
        rho = self.rho
        nearest_neiber = self.nearest_neiber

        K = np.shape(centers)[0]
        if K == 0:
            print("can not find centers")
            return

        N = np.shape(rho)[0]
        label = -1 * np.ones(N).astype(int)

        # 有几个聚类中心就分为几个簇
        for i, center in enumerate(centers):
            label[center] = i

        # 将密度从大到小排序
        index_rho = np.argsort(-rho)
        for i, point in enumerate(index_rho):

            # 从密度大的点进行标号
            if label[point] == -1:
                # 如果没有被标记过
                # 那么聚类标号与距离其最近且密度比其大的点的标号相同
                # 密度比当前点大的，一定已经先排好位置了
                label[point] = label[int(nearest_neiber[point])]
        return label