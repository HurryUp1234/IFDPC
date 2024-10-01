import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter


def compute_SC(fdpc):
    # 计算所有样本的轮廓系数
    silhouette_avg = silhouette_score(fdpc.dataset_cluster, fdpc.group_label)
    return silhouette_avg



def balance(fdpc):
    group_label = np.array(fdpc.group_label)
    data_fair = np.array(fdpc.fair_attr_vals)
    total_count = len(data_fair)

    # 定义敏感属性的集合
    sensitive_attrs_vals = set(data_fair)
    # 初始化字典，用于存储每个群集中各个敏感属性的计数
    cluster_sensitive_counts = defaultdict(lambda: defaultdict(int))
    cluster_counts = defaultdict(int)

    # 计数：全局的每个敏感属性、每个群集中的每个敏感属性和每个群集的总数
    for i in range(len(group_label)):
        cluster_counts[group_label[i]] += 1
        cluster_sensitive_counts[group_label[i]][data_fair[i]] += 1

    # 计算全局每个敏感属性的比例
    global_sensitive_proportions = {
        sensitive_attr_val: np.count_nonzero(data_fair == sensitive_attr_val) / total_count
        for sensitive_attr_val in sensitive_attrs_vals
    }

    # 初始化每个群集的balance为1（表示最大的平衡度）
    cluster_balances = defaultdict(lambda: 1.0)

    # 遍历每个群集
    for cluster_group_label, sensitive_counts in cluster_sensitive_counts.items():
        # 计算每个敏感属性的平衡度
        balances = []
        for sensitive_attr_val in sensitive_attrs_vals:
            global_sensitive_proportion = global_sensitive_proportions[sensitive_attr_val]
            cluster_sensitive_proportion = sensitive_counts[sensitive_attr_val] / cluster_counts[cluster_group_label]
            # print(sensitive_attr_val, cluster_sensitive_proportion)
            if cluster_sensitive_proportion == 0:
                balance = 0
            else:
                balance = min(cluster_sensitive_proportion /global_sensitive_proportion,
                          global_sensitive_proportion /cluster_sensitive_proportion)

            balances.append(balance)

        # 取当前簇的最小平衡度作为该簇的balance
        cluster_balances[cluster_group_label] = min(balances)

    # 计算所有群集的balance的平均值
    average_balance = np.mean(list(cluster_balances.values()))
    return average_balance
# 为何产生
def AED_AWD(fdpc):
    # 初始化 PsD 字典
    PsD = {}
    tot = len(fdpc.fair_attr_vals)

    # 计算 PsD 的分布向量
    for fair_value in fdpc.fair_attr_vals:
        PsD[fair_value] = PsD.get(fair_value, 0) + 1

    # 计算 PsD 的分布向量
    for key in PsD:
        PsD[key] = PsD[key] / tot

    # 初始化 cluster_distri 和 cluster_group_label_cnt 字典
    cluster_distri = {}
    cluster_group_label_cnt = {}

    # 计算每个簇内的敏感属性组数量和簇内点数
    for point_id, group_label in enumerate(fdpc.group_label):
        fair_value = fdpc.fair_attr_vals[point_id]

        # 更新簇内敏感属性组数量
        if group_label not in cluster_distri:
            cluster_distri[group_label] = Counter()
        cluster_distri[group_label][fair_value] += 1

        # 更新簇内点数
        if group_label not in cluster_group_label_cnt:
            cluster_group_label_cnt[group_label] = 0
        cluster_group_label_cnt[group_label] += 1

    # 计算每个簇的欧几里得距离和瓦瑟斯坦距离
    ED = {}
    WD = {}
    for group_label, distri_dict in cluster_distri.items():
        # 计算簇内的分布向量 CsD
        CsD = np.array([distri_dict[key] / cluster_group_label_cnt[group_label] for key in PsD])

        # 计算欧几里得距离和瓦瑟斯坦距离
        ED[group_label] = np.linalg.norm(list(PsD.values()) - CsD)
        WD[group_label] = wasserstein_distance(list(PsD.values()), CsD)

    # 计算平均的欧几里得距离和瓦瑟斯坦距离
    average_ED = np.mean(list(ED.values()))
    average_WD = np.mean(list(WD.values()))

    return average_ED, average_WD


def AED(fdpc):
    group_label = fdpc.group_label
    data_fair = fdpc.fair_attr_vals
    # 定义敏感属性的集合
    sensitive_attrs = set(data_fair)

    # 初始化字典，用于存储每个群集中各个敏感属性的计数
    cluster_sensitive_counts = defaultdict(lambda: defaultdict(int))
    cluster_counts = defaultdict(int)

    # 计数：全局的每个敏感属性、每个群集中的每个敏感属性和每个群集的总数
    for i in range(len(group_label)):
        cluster_counts[group_label[i]] += 1
        cluster_sensitive_counts[group_label[i]][data_fair[i]] += 1

    # 计算全局敏感属性分布向量
    global_distribution = [np.count_nonzero(data_fair == sensitive_attr) / len(data_fair) for sensitive_attr in
                           sensitive_attrs]

    # 计算每个群集的ED
    cluster_EDs = dict()
    for cluster_group_label, sensitive_counts in cluster_sensitive_counts.items():
        cluster_distribution = [sensitive_counts[sensitive_attr] / cluster_counts[cluster_group_label] for sensitive_attr in
                                sensitive_attrs]
        ed = distance.euclidean(global_distribution, cluster_distribution)
        cluster_EDs[cluster_group_label] = ed

    # 计算所有簇的平均ED
    average_ED = np.mean(list(cluster_EDs.values()))
    return average_ED



from sklearn.metrics import normalized_mutual_info_score


def get_NMI(fdpc, dpc):
    # group_label_true 和 group_label_pred 分别是原始和改进后的聚类结果标签
    return normalized_mutual_info_score(fdpc.group_label, dpc.group_label)
