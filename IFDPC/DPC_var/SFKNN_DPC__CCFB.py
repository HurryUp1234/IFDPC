# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from queue import Queue

from FDPC import get_varience
from FDPC.get_varience import read_data, cal_dc_fair_loss
from mesure.comment import balance, compute_SC


class SFKNN_DPC:
    def __init__(self, X, k=5, cluster_num=5):
        self.k = k
        self.cluster_num = cluster_num
        self.dataset_cluster = X
        self.dist_matrix = pairwise_distances(X)
        self.group_label = None
        self.centers = None
        self.weights = self.calculate_weights(X)
        self.data_name = None

    def calculate_weights(self, X):
        std_devs = np.std(X, axis=0)
        weights = std_devs / std_devs.sum()
        return weights

    def calculate_standard_deviation_weighted_distance(self):
        n, m = self.dataset_cluster.shape
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i, j] = dist_matrix[j, i] = np.sqrt(
                    np.sum(self.weights * (self.dataset_cluster[i] - self.dataset_cluster[j]) ** 2)
                )
        return dist_matrix

    def calculate_local_density_and_distance(self):
        n = self.dist_matrix.shape[0]
        rho = np.zeros(n)
        delta = np.zeros(n)

        knn_indices = np.argsort(self.dist_matrix, axis=1)[:, 1:self.k + 1]

        for i in range(n):
            rho[i] = np.sum(np.exp(-self.dist_matrix[i, knn_indices[i]]))

        max_rho_index = np.argmax(rho)
        delta[max_rho_index] = np.max(self.dist_matrix[max_rho_index])

        for i in range(n):
            if i != max_rho_index:
                higher_density_indices = np.where(rho > rho[i])[0]
                if len(higher_density_indices) > 0:
                    delta[i] = np.min(self.dist_matrix[i, higher_density_indices])
                else:
                    delta[i] = np.max(self.dist_matrix[i])


        return rho, delta, knn_indices

    def detect_density_peaks(self, rho, delta, cluster_num):
        # 计算rho和delta的乘积
        rho_delta_product = rho * delta

        # 获取rho和delta乘积的前cluster_num个点
        peaks = np.argsort(rho_delta_product)[-cluster_num:]
        self.centers = peaks
        return peaks

    def detect_outliers(self, knn_indices):
        n = self.dataset_cluster.shape[0]
        r_K = np.zeros(n)

        for i in range(n):
            r_K[i] = np.max(self.dist_matrix[i, knn_indices[i]])

        tau = np.mean(r_K)
        outliers = np.where(r_K > tau)[0]

        return outliers, tau

    def assign_non_outliers(self, rho, peaks, knn_indices, outliers):
        n = self.dist_matrix.shape[0]
        group_label = -1 * np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)

        cluster_id = 0
        for peak in peaks:
            if not visited[peak]:
                q = Queue()
                group_label[peak] = cluster_id
                visited[peak] = True
                q.put(peak)

                while not q.empty():
                    q_point = q.get()
                    for neighbor in knn_indices[q_point]:
                        if group_label[neighbor] == -1 and neighbor not in outliers:
                            theta = np.mean(self.dist_matrix[neighbor, knn_indices[neighbor]])
                            if self.dist_matrix[q_point, neighbor] < theta:
                                group_label[neighbor] = cluster_id
                                q.put(neighbor)

                cluster_id += 1

        return group_label

    def calculate_fuzzy_membership(self, i, c, knn_indices, group_label):
        neighbors = knn_indices[i]
        gamma = 1 / (1 + self.dist_matrix[i, neighbors])
        p_c_i = np.sum(gamma[group_label[neighbors] == c])
        return p_c_i

    def assign_outliers_and_unassigned(self, group_label, knn_indices):
        n = self.dataset_cluster.shape[0]
        cluster_num = len(np.unique(group_label[group_label != -1]))

        memberships = np.zeros((n, cluster_num))
        for i in range(n):
            if group_label[i] == -1:
                for c in range(cluster_num):
                    memberships[i, c] = self.calculate_fuzzy_membership(i, c, knn_indices, group_label)

        while np.max(memberships) > 0:
            s = np.unravel_index(np.argmax(memberships, axis=None), memberships.shape)[0]
            c = np.argmax(memberships[s])
            group_label[s] = c
            memberships[s, :] = 0
            for neighbor in knn_indices[s]:
                if group_label[neighbor] == -1:
                    for c in range(cluster_num):
                        memberships[neighbor, c] += self.calculate_fuzzy_membership(neighbor, c, knn_indices, group_label)

            # 如果仍有未分配的点，将它们分配到最近的聚类中心
        for i in range(n):
            if group_label[i] == -1:
                closest_center = np.argmin(self.dist_matrix[i, self.centers])
                group_label[i] = group_label[self.centers[closest_center]]
        return group_label

    def fit(self):
        self.dist_matrix = self.calculate_standard_deviation_weighted_distance()
        rho, delta, knn_indices = self.calculate_local_density_and_distance()
        peaks = self.detect_density_peaks(rho, delta, cluster_num=self.cluster_num)
        outliers, tau = self.detect_outliers(knn_indices)

        self.centers = peaks
        group_label = self.assign_non_outliers(rho, peaks, knn_indices, outliers)
        group_label = self.assign_outliers_and_unassigned(group_label, knn_indices)

        self.group_label = group_label

    def visualize_clusters(self, X, title="Clustering Results"):
        if X.shape[1] > 2:
            X = PCA(n_components=2).fit_transform(X)
        unique_group_label = np.unique(self.group_label)
        colors = np.array(["red", "blue", "green", "orange", "purple", "cyan",
                           "magenta", "beige", "hotpink", "#88c999", "black"])
        plt.figure(figsize=(10, 6))
        for i, label_id in enumerate(unique_group_label):
            if label_id == -1:
                plt.scatter(X[self.group_label == label_id, 0], X[self.group_label == label_id, 1], c='black', s=7,
                            label='Noise')
            else:
                color = colors[i % len(colors)]
                plt.scatter(X[self.group_label == label_id, 0], X[self.group_label == label_id, 1], c=color, s=7,
                            label=f'Cluster {label_id}')
        center_coords = X[self.centers] if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X[self.centers])
        plt.scatter(center_coords[:, 0], center_coords[:, 1], c='k', marker='+', s=200, label='Cluster Centers')
        plt.title(title)
        plt.legend()
        plt.show()
    def evaluate(self, data_name, k):

        self.balance_val = balance(self)
        self.sc = compute_SC(self)

            # Log to Excel
        self.log_to_excel(data_name, k)

    def log_to_excel(self, data_name, k):
        # Prepare data for Excel log
        data = {
            'Data Name': [data_name],
            'Cluster Count': [self.cluster_num],
            'K': [k],
            'SC Value': [self.sc],
            'Balance': [self.balance_val],
        }

        # Create or append to the Excel file
        excel_file = f"{data_name}_SFKNN_DPC.xlsx"
        try:
            df = pd.read_excel(excel_file)
            df = df.append(pd.DataFrame(data), ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame(data)

        # Write to Excel
        df.to_excel(excel_file, index=False)
