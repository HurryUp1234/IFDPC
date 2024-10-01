def build_tree(self):
    initial_centers = self.initial_centers
    rho = self.rho
    n = len(rho)

    pres = np.full(n, -1, dtype=int)  # ��ʼ��ÿ�����ָ��
    for center in initial_centers:
        neighbors = [c for c in initial_centers if c != center and rho[c] > rho[center]]
        if len(neighbors) > 0:
            # �ҵ��ܶȸ���������ķ� q
            closest_neighbor = min(neighbors, key=lambda x: self.deltas[center])  # ������deltaѡ������ķ�
            pres[center] = closest_neighbor
    return pres


def split_tree(self, cluster_num):
    """�������ĽṹͬʱѰ�Ҷ���ָ�㲢�����Ż���ֱ���õ�ָ�������Ĵء�"""
    # ��ʼ����
    clusters = [list(self.initial_centers)]
    while len(clusters) < cluster_num:
        best_splits = []
        # Ѱ�����д��еķָ�㣬��¼��Ϣ����
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

        # ���������Ϣ����ָ��
        if best_splits:
            # �ҵ�ȫ����ѷָ��
            best_splits.sort(key=lambda x: x[2], reverse=True)
            best_to_split = best_splits[0]  # ѡȡ��Ϣ�������ķָ�
            clusters.remove(best_to_split[0] + best_to_split[1])  # �Ƴ�ԭ��
            clusters.append(best_to_split[0])  # ����´�
            clusters.append(best_to_split[1])  # ����´�

        else:
            print(-1)

    # Ϊÿ���ط���Ψһ��ǩ
    cluster_labels = np.full(len(self.dataset_cluster), -1, dtype=int)
    for label, cluster in enumerate(clusters):
        for point in cluster:
            cluster_labels[point] = label

    return cluster_labels


def information_gain_for_split(self, parent_cluster, child1, child2):
    """����ͨ���ָ������� S1 �� S2 ����Ϣ���档"""
    initial_entropy = self.intra_cluster_entropy(parent_cluster)
    new_entropy = (self.intra_cluster_entropy(child1) * len(child1) +
                   self.intra_cluster_entropy(child2) * len(child2)) / len(parent_cluster)
    return initial_entropy - new_entropy


def intra_cluster_entropy(self, cluster):
    """������ڵ����ƶ��ء�"""
    if len(cluster) < 2:
        return 0
    indices = np.array(cluster)
    sub_matrix = self.A[np.ix_(indices, indices)]
    intra_similarity = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
    return np.std(intra_similarity)


    def outliers_searching(self):
        # **��Ⱥ����**
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
        ʹ���������ճ��ӷ��Ż���ָʾ������󻯴ص�ƽ�����ƶ�֮�͡�
        """
        m = len(self.initial_centers)  # �ܶȷ������
        k = cluster_num  # ָ���Ĵ�����
        MC_k = np.random.rand(m, k)  # �����ʼ����ָʾ����
        lambda_vals = np.zeros(m)  # �������ճ��ӳ�ʼ��
        eta = 0.01  # ѧϰ��
        A = self.A  # ȫ�����ƶȾ���
        MC_k_old = MC_k.copy()

        for iteration in range(max_iterations):
            # ����ÿ���ص�ƽ�����ƶ� SA
            for j in range(k):
                cluster_indices = np.where(MC_k[:, j] > 0.5)[0]  # ��ǰ�ص������ܶȷ�
                cluster_size = len(cluster_indices)
                if cluster_size > 1:
                    SA_j = np.sum(A[np.ix_(cluster_indices, cluster_indices)]) / (cluster_size * (cluster_size - 1))
                else:
                    SA_j = 0  # ������ʱ��ƽ�����ƶȶ���Ϊ0

                # ���� MC_k ������� SA_j
                for i in range(m):
                    MC_k[i, j] += eta * SA_j  # ���´�ָʾ����ʹ�䳯�����ƽ�����ƶȷ����ƶ�

            # ͶӰ��[0, 1]����
            MC_k = np.clip(MC_k, 0, 1)

            # �����������ճ��ӣ�ȷ��ÿ���ܶȷ�ֻ����һ����
            for i in range(m):
                lambda_vals[i] += eta * (1 - np.sum(MC_k[i, :]))

            # �ж���������
            if np.linalg.norm(MC_k - MC_k_old) < epsilon:
                print(f"Converged in {iteration} iterations")
                break

            MC_k_old = MC_k.copy()

        # �������ݴ�ָʾ�������ر�ǩ
        return self.assign_clusters(MC_k)

    def assign_clusters(self, MC_k):
        """
        ���ݴ�ָʾ����MC_k���������յĴر�ǩ��������ǩ���ܶȷ�ӳ�䵽ԭʼ���ݼ������е�ı�š�
        """
        # ��ȡÿ���ܶȷ�Ĵر�ǩ
        center_labels = np.argmax(MC_k, axis=1)

        # ��ʼ���������ݵ�����մر�ǩΪ-1
        final_labels = -1 * np.ones(self.n)

        # ���ܶȷ�Ĵر�ǩӳ�䵽ԭʼ���ݼ��еĶ�Ӧ��
        for idx, center in enumerate(self.initial_centers):
            final_labels[center] = center_labels[idx]

        return final_labels
    def compute_average_similarity(self, nodes):
        """
        ����ڵ��б������нڵ��֮���ƽ�����ƶȡ�
        nodes: �������нڵ��ŵ��б�
        """
        similarity_sum = 0
        count = 0
        # ����ڵ��֮������ƶ�
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity_sum += self.A[nodes[i], nodes[j]]  # ʹ��ȫ�����ƶȾ���A
                count += 1

        if count > 0:
            return similarity_sum / count
        else:
            return 0  # ���û�нڵ�ԣ����ƶ�Ϊ0

    def find_best_split(self, tree, k):
        """
        �ݹ�طָ������ҵ����� k ���ڵ����ѷָ�㣬��󻯷ָ����������ƶȺ͡�

        :param tree: ���ĸ��ڵ�
        :param k: ÿ�ηָ���������а����Ľڵ���
        :return: ���ŷָ���������Ϻ������ƶȺ�
        """
        # Step 1: �Ӹ��ڵ�������ҵ���� >= k �ķָ�ڵ�
        CP = self.find_split_nodes(tree, k)

        if len(CP) < k:
            return None  # ����Ҳ����㹻�Ľڵ㣬��ֹͣ�ݹ�

        # Step 2: ����ָ����������ƶ�
        P_CP, P_CP_similarity = [], []
        for node in CP:
            P_CP.append(tree[node].subtree_nodes)
            P_CP_similarity.append(tree[node].average_similarity)

        # ����ʣ���������ƶ�
        remaining_nodes = []
        for node in tree:
            if node not in P_CP:
                remaining_nodes.append(node)
        remaining_tree_similarity = self.compute_average_similarity(remaining_nodes)

        # ��¼���ƶȺ�
        total_similarity = remaining_tree_similarity + sum(P_CP_similarity)

        # Step 3: ѡ����ʹ SUM(P_CP_i) ��󻯵ķָ��
        best_split_point = max(P_CP_similarity)

        # Step 4: �ݹ�����·ָ�
        return self.find_best_split(tree, k)

    def find_split_nodes(self, tree, k):
        """
        �����ĸ��ڵ�������ҵ��ܹ��������� k ���ڵ�ķָ�㡣
        :param tree: ���ĸ��ڵ�
        :param k: �ָ���Ľڵ�����
        :return: �ָ�ڵ㼯��
        """
        # ʵ�ִӸ���ʼ����Ѱ�Ұ�������k���ڵ�����
        nodes_at_depth = []
        # �ݹ���������ҵ����������ķָ�ڵ�
        # ���طָ�㼯�� CP
        pass
    def build_tree(self):
        initial_centers = self.initial_centers
        rho = self.rho
        n = len(rho)

        pres = np.full(n, -1, dtype=int)  # ��ʼ��ÿ�����ָ��
        tree = {center: TreeNode(center) for center in initial_centers}  # �������ڵ�

        for center in initial_centers:
            neighbors = [c for c in initial_centers if c != center and rho[c] > rho[center]]
            if len(neighbors) > 0:
                # �ҵ��ܶȸ���������ķ� q
                closest_neighbor_index = np.argmin(self.dist_matrix[center, neighbors])
                fa = neighbors[closest_neighbor_index]
                pres[center] = fa  # ���ø��ڵ�

        # ����ÿ���ڵ��ƽ�����ƶ�
        return tree, pres


    def get_subtree_nodes(self, node, pres):
        subtree = [node]  # ��ʼ����������������
        children = [i for i, p in enumerate(pres) if p == node]  # �ҵ�����ָ��ýڵ���ӽڵ�

        for child in children:
            subtree.extend(self.get_subtree_nodes(child, pres))  # �ݹ��ȡ�ӽڵ�������ڵ�

        return subtree


    def adjust_clusters(self, delta=0.4):


        clusters = self.group_label
        n = len(clusters)
        updated_clusters = clusters.copy()

        # Step 1: �ҵ�ÿ����������
        nearest_points = np.full(n, -1, dtype=int)
        for i in range(n):
            valid_centers = [center for center in self.initial_centers if center != i]
            nearest_points[i] = valid_centers[np.argmin(self.dist_matrix[i, valid_centers])]

        # Step 2: ������Ϊ����������ڲ�ͬ�صķ��
        for i in self.initial_centers:
            j = nearest_points[i]  # i ������� j
            # print(i, j, self.group_label[i], self.group_label[j])
            if nearest_points[j] == i and updated_clusters[i] != updated_clusters[j]:
                similarity = self.A[i, j]
                if similarity > delta:
                    print(1)
                    # �ҵ�ÿ�������Լ���������ķ�
                    nearest_in_cluster_i = self.find_nearest_in_cluster(i, updated_clusters)
                    nearest_in_cluster_j = self.find_nearest_in_cluster(j, updated_clusters)

                    if nearest_in_cluster_j == None:
                        updated_clusters[j] = updated_clusters[i]
                        continue
                    if nearest_in_cluster_i == None:
                        updated_clusters[i] = updated_clusters[j]
                        continue
                    # �Ƚϴ�����������
                    if self.dist_matrix[i, nearest_in_cluster_i] > self.dist_matrix[j, nearest_in_cluster_j]:
                        updated_clusters[i] = updated_clusters[j]
                    else:
                        updated_clusters[j] = updated_clusters[i]

        return updated_clusters

    def assign_remaining_points(self):
        for i in range(self.n):
            if self.group_label[i] == -1:  # ����õ�δ�����䵽�κδ�
                # ����� i ������ initial_centers �ľ���
                nearest_center = self.peaks[np.argmin(self.nor_dis[i, self.peaks])]
                # ���õ���������� initial_centers �Ĵ�
                self.group_label[i] = self.group_label[nearest_center]

    def assign_remaining_points(self):
        # Step 1: ��ʼ��ģ�������Ⱦ���
        unassigned_points = np.where(self.group_label == -1)[0]
        membership_matrix = np.zeros((len(unassigned_points), len(self.peaks)))

        # ��ÿ��δ����ĵ㣬����������ܶȷ�ľ��룬����ʼ��������
        for idx, point_idx in enumerate(unassigned_points):
            for c_idx, center_idx in enumerate(self.peaks):
                dist = self.dist_matrix[point_idx, center_idx]
                membership_matrix[idx, c_idx] = 1 / (1 + dist)  # ��ʼ��������Ϊ����ĵ���

        # Step 2: �����㷨���̽��е���
        while np.max(membership_matrix) > 0:
            # �ҵ�������������ȵĵ㼰���Ӧ�Ĵ�
            max_idx = np.unravel_index(np.argmax(membership_matrix, axis=None), membership_matrix.shape)
            point_idx = unassigned_points[max_idx[0]]  # ��ȡ�õ������
            cluster_idx = max_idx[1]  # ��ȡ�õ������ڵĴ�

            # ����������ƶȵ����
            self.group_label[point_idx] = cluster_idx

            # ��������������ȣ������ظ�ѡ��
            membership_matrix[max_idx[0], :] = 0

            def calculate_gamma(q, s):
                w_ij = self.dist_matrix[q, s]
                sum_w_lj = np.sum([self.dist_matrix[l, s] for l in self.KNN[s]])
                # ���������
                if sum_w_lj == 0:
                    return 0
                gamma_ij = w_ij / sum_w_lj
                return gamma_ij

            # ���¸õ��K���ڵ��������
            neighbors = self.KNN[point_idx]
            for neighbor_idx in neighbors:
                if self.group_label[neighbor_idx] == -1:  # ������δ����ĵ�
                    for c_idx in range(len(self.peaks)):
                        sim = self.A[point_idx, neighbor_idx]
                        gamma = calculate_gamma(neighbor_idx, point_idx)
                        membership_matrix[unassigned_points == neighbor_idx, c_idx] += gamma * sim

        return self.group_label


    def assign_remaining_points(self):




        # ��ȡδ����ĵ�
        unassigned_points = np.where(self.group_label == -1)[0]

        # ��ʼ��������ǩ����
        similarity_based_labels = -1 * np.ones(len(unassigned_points), dtype=int)
        distance_based_labels = -1 * np.ones(len(unassigned_points), dtype=int)

        # ��һ�������������Է���
        for idx, point_idx in enumerate(unassigned_points):
            max_similarity = -np.inf
            best_peak = -1
            for peak_idx in self.peaks:
                # ���������ԣ�������Ը���Ȩ�ػ��߾���ĵ��������㣩
                similarity = self.A[point_idx, peak_idx]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_peak = peak_idx
            similarity_based_labels[idx] = best_peak

        # �ڶ�������������Է���
        for idx, point_idx in enumerate(unassigned_points):
            min_distance = np.inf
            best_peak = -1
            for peak_idx in self.peaks:
                # ֱ��ʹ�þ���
                distance = self.dist_matrix[point_idx, peak_idx]
                if distance < min_distance:
                    min_distance = distance
                    best_peak = peak_idx
            distance_based_labels[idx] = best_peak

        # ���������Ƚ����ֱ�ǩ�������������ͬ��ȷ�ϱ�ǩ
        for idx, point_idx in enumerate(unassigned_points):
            if similarity_based_labels[idx] == distance_based_labels[idx]:
                self.group_label[point_idx] = similarity_based_labels[idx]

        return self.group_label




    def get_newDis(self, alpha):
        dists = self.dist_matrix
        # ���������Dӳ�䵽0-1����
        D_min = np.min(dists)
        D_max = np.max(dists)
        normalized_dists = (dists - D_min) / (D_max - D_min)
        # ��Ȩ�������: D_weighted = D + alpha * A
        weighted_dists = (1 - alpha) * normalized_dists + alpha * -self.A

        return weighted_dists


    # ��Ȼ�ھ�����ֵ
    def NaN_Searching(self):
        # ��ʼ��������Χr�ͷ������������nb
        r = 1
        n = self.dataset_cluster.shape[0]
        nb = np.zeros(n, dtype=int)  # ��������ڼ���

        # ��ʼʱû�з�������ڵĵ�����
        Num = [n]

        while True:
            # �����е��������ҵ�r���ھ�
            for p in range(n):
                # ����p�ĵ�r�������q
                nearest_neighbors = np.argsort(self.dist_matrix[p])
                if r < len(nearest_neighbors):
                    q = nearest_neighbors[r]
                    # ����q�ķ�������ڼ���
                    nb[q] += 1

            # ͳ�Ʒ����������Ϊ0�ĵ�
            Num.append(np.sum(nb == 0))

            # �ж��Ƿ�������ֹ�����������������Ϊ0�ĵ��������ٱ仯
            if Num[-1] == Num[-2]:
                break

            # ����������Χ
            r += 1

        # ������Ȼ�ھ�����ֵ���������뾶r
        return r, nb