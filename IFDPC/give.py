def build_tree(self):
    initial_centers = self.initial_centers
    rho = self.rho
    n = len(rho)

    pres = np.full(n, -1, dtype=int)  # 初始化每个峰的指向
    for center in initial_centers:
        neighbors = [c for c in initial_centers if c != center and rho[c] > rho[center]]
        if len(neighbors) > 0:
            # 找到密度更大且最近的峰 q
            closest_neighbor = min(neighbors, key=lambda x: self.deltas[center])  # 按距离delta选择最近的峰
            pres[center] = closest_neighbor
    return pres


def split_tree(self, cluster_num):
    """利用树的结构同时寻找多个分割点并迭代优化，直到得到指定数量的簇。"""
    # 初始化簇
    clusters = [list(self.initial_centers)]
    while len(clusters) < cluster_num:
        best_splits = []
        # 寻找所有簇中的分割点，记录信息增益
        for cluster in clusters:
            best_gain = -np.inf
            best_split = None
            for i in range(1, len(cluster)):
                child1 = cluster[:i]
                child2 = cluster[i:]
                gain = self.information_gain_for_split(cluster, child1, child2)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (child1, child2)
            if best_split:
                best_splits.append((best_split[0], best_split[1], best_gain))

        # 根据最佳信息增益分割簇
        if best_splits:
            # 找到全局最佳分割点
            best_splits.sort(key=lambda x: x[2], reverse=True)
            best_to_split = best_splits[0]  # 选取信息增益最大的分割
            clusters.remove(best_to_split[0] + best_to_split[1])  # 移除原簇
            clusters.append(best_to_split[0])  # 添加新簇
            clusters.append(best_to_split[1])  # 添加新簇

        else:
            print(-1)

    # 为每个簇分配唯一标签
    cluster_labels = np.full(len(self.dataset_cluster), -1, dtype=int)
    for label, cluster in enumerate(clusters):
        for point in cluster:
            cluster_labels[point] = label

    return cluster_labels


def information_gain_for_split(self, parent_cluster, child1, child2):
    """计算通过分割两个簇 S1 和 S2 的信息增益。"""
    initial_entropy = self.intra_cluster_entropy(parent_cluster)
    new_entropy = (self.intra_cluster_entropy(child1) * len(child1) +
                   self.intra_cluster_entropy(child2) * len(child2)) / len(parent_cluster)
    return initial_entropy - new_entropy


def intra_cluster_entropy(self, cluster):
    """计算簇内的相似度熵。"""
    if len(cluster) < 2:
        return 0
    indices = np.array(cluster)
    sub_matrix = self.A[np.ix_(indices, indices)]
    intra_similarity = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
    return np.std(intra_similarity)


    def outliers_searching(self):
        # **离群点检测**
        n = self.dataset_cluster.shape[0]
        rho = self.rho
        dists = self.dist_matrix
        r_d = np.zeros(n)
        r_rho = np.zeros(n)
        # Compute r_i^K and densities
        for i in range(n):
            r_d[i] = np.max(self.deltas[self.KNN[i]])  # Max distance to K-nearest neighbor
            r_rho[i] = np.min(rho[self.KNN[i]])

        avg_d = np.mean(r_d)

        # Determine low-density points
        avg_density = np.mean(r_rho)
        low_density_points = r_rho < avg_density

        # Determine high relative distance points
        high_distance_points = r_d > avg_d

        # Combine both criteria low_density_points & high_distance_points
        outliers = np.where(high_distance_points)[0]
        return outliers


    def cluster_with_lagrange(self, cluster_num, max_iterations=300, epsilon=0.01, beta=0.5):
        """
        使用拉格朗日乘子法优化簇指示矩阵，最大化簇的平均相似度之和。
        """
        m = len(self.initial_centers)  # 密度峰的数量
        k = cluster_num  # 指定的簇数量
        MC_k = np.random.rand(m, k)  # 随机初始化簇指示矩阵
        lambda_vals = np.zeros(m)  # 拉格朗日乘子初始化
        eta = 0.01  # 学习率
        A = self.A  # 全局相似度矩阵
        MC_k_old = MC_k.copy()

        for iteration in range(max_iterations):
            # 计算每个簇的平均相似度 SA
            for j in range(k):
                cluster_indices = np.where(MC_k[:, j] > 0.5)[0]  # 当前簇的所有密度峰
                cluster_size = len(cluster_indices)
                if cluster_size > 1:
                    SA_j = np.sum(A[np.ix_(cluster_indices, cluster_indices)]) / (cluster_size * (cluster_size - 1))
                else:
                    SA_j = 0  # 单个点时，平均相似度定义为0

                # 更新 MC_k 矩阵，最大化 SA_j
                for i in range(m):
                    MC_k[i, j] += eta * SA_j  # 更新簇指示矩阵，使其朝向最大化平均相似度方向移动

            # 投影到[0, 1]区间
            MC_k = np.clip(MC_k, 0, 1)

            # 更新拉格朗日乘子，确保每个密度峰只属于一个簇
            for i in range(m):
                lambda_vals[i] += eta * (1 - np.sum(MC_k[i, :]))

            # 判断收敛条件
            if np.linalg.norm(MC_k - MC_k_old) < epsilon:
                print(f"Converged in {iteration} iterations")
                break

            MC_k_old = MC_k.copy()

        # 后处理：根据簇指示矩阵分配簇标签
        return self.assign_clusters(MC_k)

    def assign_clusters(self, MC_k):
        """
        根据簇指示矩阵MC_k，分配最终的簇标签，并将标签从密度峰映射到原始数据集中所有点的编号。
        """
        # 获取每个密度峰的簇标签
        center_labels = np.argmax(MC_k, axis=1)

        # 初始化所有数据点的最终簇标签为-1
        final_labels = -1 * np.ones(self.n)

        # 将密度峰的簇标签映射到原始数据集中的对应点
        for idx, center in enumerate(self.initial_centers):
            final_labels[center] = center_labels[idx]

        return final_labels
    def compute_average_similarity(self, nodes):
        """
        计算节点列表中所有节点对之间的平均相似度。
        nodes: 包含所有节点编号的列表
        """
        similarity_sum = 0
        count = 0
        # 计算节点对之间的相似度
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity_sum += self.A[nodes[i], nodes[j]]  # 使用全局相似度矩阵A
                count += 1

        if count > 0:
            return similarity_sum / count
        else:
            return 0  # 如果没有节点对，相似度为0

    def find_best_split(self, tree, k):
        """
        递归地分割树，找到包含 k 个节点的最佳分割点，最大化分割后的子树相似度和。

        :param tree: 树的根节点
        :param k: 每次分割出的子树中包含的节点数
        :return: 最优分割的子树集合和其相似度和
        """
        # Step 1: 从根节点出发，找到深度 >= k 的分割节点
        CP = self.find_split_nodes(tree, k)

        if len(CP) < k:
            return None  # 如果找不到足够的节点，则停止递归

        # Step 2: 计算分割后的子树相似度
        P_CP, P_CP_similarity = [], []
        for node in CP:
            P_CP.append(tree[node].subtree_nodes)
            P_CP_similarity.append(tree[node].average_similarity)

        # 计算剩余树的相似度
        remaining_nodes = []
        for node in tree:
            if node not in P_CP:
                remaining_nodes.append(node)
        remaining_tree_similarity = self.compute_average_similarity(remaining_nodes)

        # 记录相似度和
        total_similarity = remaining_tree_similarity + sum(P_CP_similarity)

        # Step 3: 选择能使 SUM(P_CP_i) 最大化的分割点
        best_split_point = max(P_CP_similarity)

        # Step 4: 递归地向下分割
        return self.find_best_split(tree, k)

    def find_split_nodes(self, tree, k):
        """
        从树的根节点出发，找到能够包含至少 k 个节点的分割点。
        :param tree: 树的根节点
        :param k: 分割出的节点数量
        :return: 分割节点集合
        """
        # 实现从根开始向下寻找包含至少k个节点的深度
        nodes_at_depth = []
        # 递归遍历树，找到满足条件的分割节点
        # 返回分割点集合 CP
        pass
    def build_tree(self):
        initial_centers = self.initial_centers
        rho = self.rho
        n = len(rho)

        pres = np.full(n, -1, dtype=int)  # 初始化每个峰的指向
        tree = {center: TreeNode(center) for center in initial_centers}  # 创建树节点

        for center in initial_centers:
            neighbors = [c for c in initial_centers if c != center and rho[c] > rho[center]]
            if len(neighbors) > 0:
                # 找到密度更大且最近的峰 q
                closest_neighbor_index = np.argmin(self.dist_matrix[center, neighbors])
                fa = neighbors[closest_neighbor_index]
                pres[center] = fa  # 设置父节点

        # 计算每个节点的平均相似度
        return tree, pres


    def get_subtree_nodes(self, node, pres):
        subtree = [node]  # 初始化，子树包括自身
        children = [i for i, p in enumerate(pres) if p == node]  # 找到所有指向该节点的子节点

        for child in children:
            subtree.extend(self.get_subtree_nodes(child, pres))  # 递归获取子节点的子树节点

        return subtree


    def adjust_clusters(self, delta=0.4):


        clusters = self.group_label
        n = len(clusters)
        updated_clusters = clusters.copy()

        # Step 1: 找到每个点的最近峰
        nearest_points = np.full(n, -1, dtype=int)
        for i in range(n):
            valid_centers = [center for center in self.initial_centers if center != i]
            nearest_points[i] = valid_centers[np.argmin(self.dist_matrix[i, valid_centers])]

        # Step 2: 遍历互为最近峰且属于不同簇的峰对
        for i in self.initial_centers:
            j = nearest_points[i]  # i 的最近峰 j
            # print(i, j, self.group_label[i], self.group_label[j])
            if nearest_points[j] == i and updated_clusters[i] != updated_clusters[j]:
                similarity = self.A[i, j]
                if similarity > delta:
                    print(1)
                    # 找到每个点在自己簇内最近的峰
                    nearest_in_cluster_i = self.find_nearest_in_cluster(i, updated_clusters)
                    nearest_in_cluster_j = self.find_nearest_in_cluster(j, updated_clusters)

                    if nearest_in_cluster_j == None:
                        updated_clusters[j] = updated_clusters[i]
                        continue
                    if nearest_in_cluster_i == None:
                        updated_clusters[i] = updated_clusters[j]
                        continue
                    # 比较簇内最近点距离
                    if self.dist_matrix[i, nearest_in_cluster_i] > self.dist_matrix[j, nearest_in_cluster_j]:
                        updated_clusters[i] = updated_clusters[j]
                    else:
                        updated_clusters[j] = updated_clusters[i]

        return updated_clusters

    def assign_remaining_points(self):
        for i in range(self.n):
            if self.group_label[i] == -1:  # 如果该点未被分配到任何簇
                # 计算点 i 到所有 initial_centers 的距离
                nearest_center = self.peaks[np.argmin(self.nor_dis[i, self.peaks])]
                # 将该点分配给最近的 initial_centers 的簇
                self.group_label[i] = self.group_label[nearest_center]

    def assign_remaining_points(self):
        # Step 1: 初始化模糊隶属度矩阵
        unassigned_points = np.where(self.group_label == -1)[0]
        membership_matrix = np.zeros((len(unassigned_points), len(self.peaks)))

        # 对每个未分配的点，计算与最近密度峰的距离，并初始化隶属度
        for idx, point_idx in enumerate(unassigned_points):
            for c_idx, center_idx in enumerate(self.peaks):
                dist = self.dist_matrix[point_idx, center_idx]
                membership_matrix[idx, c_idx] = 1 / (1 + dist)  # 初始化隶属度为距离的倒数

        # Step 2: 按照算法流程进行迭代
        while np.max(membership_matrix) > 0:
            # 找到具有最大隶属度的点及其对应的簇
            max_idx = np.unravel_index(np.argmax(membership_matrix, axis=None), membership_matrix.shape)
            point_idx = unassigned_points[max_idx[0]]  # 获取该点的索引
            cluster_idx = max_idx[1]  # 获取该点隶属于的簇

            # 矩阵最大相似度点分配
            self.group_label[point_idx] = cluster_idx

            # 清除这个点的隶属度，避免重复选择
            membership_matrix[max_idx[0], :] = 0

            def calculate_gamma(q, s):
                w_ij = self.dist_matrix[q, s]
                sum_w_lj = np.sum([self.dist_matrix[l, s] for l in self.KNN[s]])
                # 避免除以零
                if sum_w_lj == 0:
                    return 0
                gamma_ij = w_ij / sum_w_lj
                return gamma_ij

            # 更新该点的K近邻点的隶属度
            neighbors = self.KNN[point_idx]
            for neighbor_idx in neighbors:
                if self.group_label[neighbor_idx] == -1:  # 仅更新未分配的点
                    for c_idx in range(len(self.peaks)):
                        sim = self.A[point_idx, neighbor_idx]
                        gamma = calculate_gamma(neighbor_idx, point_idx)
                        membership_matrix[unassigned_points == neighbor_idx, c_idx] += gamma * sim

        return self.group_label


    def assign_remaining_points(self):




        # 获取未分配的点
        unassigned_points = np.where(self.group_label == -1)[0]

        # 初始化两个标签数组
        similarity_based_labels = -1 * np.ones(len(unassigned_points), dtype=int)
        distance_based_labels = -1 * np.ones(len(unassigned_points), dtype=int)

        # 第一步：基于相似性分配
        for idx, point_idx in enumerate(unassigned_points):
            max_similarity = -np.inf
            best_peak = -1
            for peak_idx in self.peaks:
                # 计算相似性（这里可以根据权重或者距离的倒数来计算）
                similarity = self.A[point_idx, peak_idx]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_peak = peak_idx
            similarity_based_labels[idx] = best_peak

        # 第二步：基于最近性分配
        for idx, point_idx in enumerate(unassigned_points):
            min_distance = np.inf
            best_peak = -1
            for peak_idx in self.peaks:
                # 直接使用距离
                distance = self.dist_matrix[point_idx, peak_idx]
                if distance < min_distance:
                    min_distance = distance
                    best_peak = peak_idx
            distance_based_labels[idx] = best_peak

        # 第三步：比较两种标签分配结果，如果相同则确认标签
        for idx, point_idx in enumerate(unassigned_points):
            if similarity_based_labels[idx] == distance_based_labels[idx]:
                self.group_label[point_idx] = similarity_based_labels[idx]

        return self.group_label




    def get_newDis(self, alpha):
        dists = self.dist_matrix
        # 将距离矩阵D映射到0-1区间
        D_min = np.min(dists)
        D_max = np.max(dists)
        normalized_dists = (dists - D_min) / (D_max - D_min)
        # 加权距离矩阵: D_weighted = D + alpha * A
        weighted_dists = (1 - alpha) * normalized_dists + alpha * -self.A

        return weighted_dists


    # 自然邻居特征值
    def NaN_Searching(self):
        # 初始化搜索范围r和反向最近邻数量nb
        r = 1
        n = self.dataset_cluster.shape[0]
        nb = np.zeros(n, dtype=int)  # 反向最近邻计数

        # 初始时没有反向最近邻的点数量
        Num = [n]

        while True:
            # 逐点进行迭代，查找第r个邻居
            for p in range(n):
                # 查找p的第r个最近邻q
                nearest_neighbors = np.argsort(self.dist_matrix[p])
                if r < len(nearest_neighbors):
                    q = nearest_neighbors[r]
                    # 更新q的反向最近邻计数
                    nb[q] += 1

            # 统计反向最近邻数为0的点
            Num.append(np.sum(nb == 0))

            # 判断是否满足终止条件：反向最近邻数为0的点数量不再变化
            if Num[-1] == Num[-2]:
                break

            # 增加搜索范围
            r += 1

        # 返回自然邻居特征值，即搜索半径r
        return r, nb