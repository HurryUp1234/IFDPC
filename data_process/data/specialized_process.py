# 各个数据集的单独处理办法
import networkx as nx
import numpy as np
import pandas as pd
import random
from io import StringIO
from sklearn.metrics import euclidean_distances

pd.options.mode.chained_assignment = None  # default='warn '
project_path = 'D:\\env\\py_all\\cluster\\FDPC-main\\FDPC-main\\data_process\\data\\fairness_data\\'
project_path2 = 'D:\\env\\py_all\\cluster\\FDPC-main\\FDPC-main\\data_process\\data\\dataset\\'


def create_graph_from_data(data, threshold):
    G = nx.Graph()
    distances = euclidean_distances(data)
    num_points = data.shape[0]

    # 将距离矩阵转换为一维数组，并计算给定百分位数的距离阈值
    dist_array = distances[np.triu_indices(num_points, k=1)]
    threshold = np.percentile(dist_array, threshold)

    for i in range(num_points):
        # print(i)
        for j in range(i + 1, num_points):
            if distances[i][j] < threshold:
                G.add_edge(i + 1, j + 1)
    return G


def extract_largest_connected_subgraph(G, data):
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph_data = data.loc[list(largest_cc)]
    return subgraph_data


def get_data_glass():
    attr_string = 'RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,class'

    data = pd.read_csv(
        project_path2 + 'glass.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'class'

    # 随机样本数

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_seeds():
    attr_string = 'area,perimeter,compactness,length of kernel,width of kernel,asymmetry coefficient,length of kernel groove,class'

    data = pd.read_csv(
        project_path2 + 'seeds.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'class'

    # 随机样本数

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_wdbc():
    attr_string = 'radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,' \
                  'concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_SE,texture_SE,' \
                  'perimeter_SE,area_SE,moothness_SE,compactness_SE,concavity_SE,concave points_SE,symmetry_SE,fractal_dimension_SE,' \
                  'radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,' \
                  'concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst,class'

    data = pd.read_csv(
        project_path2 + 'wdbc.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'class'
    # 非数字型属性处理
    sex_mapping = {' M': 0, ' B': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(sex_mapping)
    # 随机样本数

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_wine():
    attr_string = 'Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,' \
                  'Nonflavanoid phenols,Proanthocyanins,' \
                  'Color intensity,Hue,OD280/OD315 of diluted wines,Proline ,class'

    data = pd.read_csv(
        project_path2 + 'wine.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'class'

    # 随机样本数

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_iris():
    # 加载数据
    data = pd.read_csv(
        project_path2 + 'iris.csv')
    # 聚类属性和敏感属性
    attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    # 敏感属性
    fair_attr = 'class'
    # 提取指定的属性列
    selected_data = data[attributes]
    # 非数字型属性处理
    sex_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    selected_data['class'] = selected_data['class'].map(sex_mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_adult():
    # 加载数据
    data = pd.read_csv(
        project_path + 'adult_processed.csv'  # adult.data
    )
    # print(data.columns)
    # 聚类属性和敏感属性
    atrr_string = "age,education-num,hours-per-week,sex"  # race,
    attributes = atrr_string.split(',')
    # print(attributes)
    # 提取指定的属性列
    selected_data = data[attributes]
    # 非数字型属性处理
    # fair_attr_mapping = {
    #     "White": 0,
    #     "Black": 1,
    #     "Asian-Pac-Islander": 2,
    #     "Amer-Indian-Eskimo": 3,
    #     "Other": 4
    # }
    fair_attr = 'sex'
    # fair_attr_mapping = {' Female': 0, ' Male': 1}
    # selected_data[fair_attr] = selected_data[fair_attr].map(fair_attr_mapping).fillna(-1).astype(int)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_bank():
    # 加载数据
    data = pd.read_csv(
        project_path + 'bank_processed.csv', # _processed
        # delimiter=';', quotechar='"'
    )
    # 聚类属性和敏感属性
    attr_str = 'age,balance,duration,balance,marital'
    attributes = attr_str.split(',')  # 'education'
    # 敏感属性
    # fair_attr = 'education' #
    fair_attr = 'marital'
    # 提取指定的属性列
    selected_data = data[attributes]
    # 非数字型属性处理
    # marital_mapping = {'single': 0, 'married': 1, 'divorced': 2}
    # education_mapping = {
    #     'primary': 0,
    #     'secondary': 1,
    #     'tertiary': 2
    # }

    # selected_data.loc[:, fair_attr] = selected_data[fair_attr].map(marital_mapping).fillna(3).astype(int)
    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_obesity():
    # 加载数据
    data = pd.read_csv(
        project_path + 'obesity.csv')
    # 聚类属性和敏感属性
    attributes = ['Age', 'Height', 'Weight', 'Gender']
    # 敏感属性
    fair_attr = 'Gender'
    # 提取指定的属性列
    selected_data = data[attributes]
    # 非数字型属性处理
    sex_mapping = {'Female': 0, 'Male': 1}
    selected_data['Gender'] = selected_data['Gender'].map(sex_mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_census1990():
    # 加载数据
    data = pd.read_csv(
        project_path + 'census1990.csv')
    # 聚类属性和敏感属性
    attributes = ['dAncstry1', 'dAncstry2', 'iAvail', 'iCitizen', 'iClass', 'dDepart', 'iSex']
    # 敏感属性
    fair_attr = 'iSex'
    # 提取指定的属性列
    selected_data = data[attributes]
    num_samples = 2000
    selected_data = selected_data.sample(num_samples, random_state=None)
    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_creditcard():
    # 加载数据
    data = pd.read_csv(
         project_path + 'creditcard.csv')
    # 给定的属性字符串
    data_string = "LIMIT_BAL,EDUCATION,BILL_AMT1,BILL_AMT2," \
                  "BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6"

    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = data_string.split(',')
    # 敏感属性
    fair_attr = 'EDUCATION'  # SEX
    # 提取指定的属性列
    selected_data = data[attributes]
    # 随机样本数
    num_samples = 2000
    selected_data = selected_data.sample(num_samples, random_state=None)
    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1

    return selected_data, fair_attr


def get_data_drug_consumption():
    # 加载数据
    attr_string = "ID,Age,Gender,Education,Country,Ethnicity,Nscore,Escore,Oscore,Ascore,Cscore,Impulsive,SS"
    data = pd.read_csv(
        project_path + 'drug_consumption.data')

    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 敏感属性
    fair_attr = 'Gender'  # gender # country # ethnicity # education
    # 提取指定的属性列
    selected_data = data[attributes]

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


# 失败
def get_data_drug():
    # 加载数据
    data = pd.read_csv(
        project_path + 'drug.csv',
    )

    # 给定的属性字符串
    data_string = "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28," \
                  "29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47," \
                  "85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102," \
                  "103,104,105,106,107,108,109,110,112,113,114,115,116,117,118,119,120,121," \
                  "122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140," \
                  "141,142,143,144,145,146,147,148,149,150,151,152,153,154,Ethnicity,Gender,HasTie"
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = data_string.split(',')
    # 敏感属性
    fair_attr = 'HasTie'  # Gender # country # Ethnicity # education # HasTie
    # 提取指定的属性列
    selected_data = data[attributes]

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_dabestic():
    # 加载数据
    data = pd.read_csv(
        project_path + 'dabestic.csv'
        , delimiter='\t')
    # data 是包含制表符分隔的数据的字符串
    # 聚类属性和敏感属性
    attributes = ['AGE', 'SEX', 'BMI', 'S1', 'S2', 'S3', 'BP', 'S6']
    # 敏感属性
    fair_attr = 'SEX'
    # 提取指定的属性列
    selected_data = data[attributes]

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_hcvdat0():
    data = pd.read_csv(
        project_path + 'hcvdat0.csv',
    )
    # print(data.columns)
    # 聚类属性和敏感属性
    atrr_string = "Sex,AST,BIL,CHE,CREA,GGT"
    attributes = atrr_string.split(',')
    # print(attributes)
    # 敏感属性
    fair_attr = 'Sex'
    # 提取指定的属性列
    selected_data = data[attributes]
    # 非数字型属性处理
    season_mapping = {'m': 0, 'f': 1}
    selected_data.loc[:, fair_attr] = selected_data[fair_attr].map(season_mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_athlete():
    # 加载数据
    data = pd.read_csv(
        project_path + 'athlete_processed.csv'
        , quotechar='"'
    )
    # 替换所有的NaN值为-1
    data = data.fillna(-1)
    # print(data.columns)
    # 聚类属性和敏感属性
    atrr_string = "Age,Height,Weight,Year,Season"
    # Sex,
    attributes = atrr_string.split(',')
    # print(attributes)
    # 敏感属性
    fair_attr = 'Season'
    # 提取指定的属性列
    selected_data = data[attributes]
    # 非数字型属性处理
    # season_mapping = {'M': 0, 'F': 1}
    # season_mapping = {'Summer': 0, 'Winter': 1}
    # selected_data.loc[:, fair_attr] = selected_data[fair_attr].map(season_mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : 编号为1的点的'class'属性值
    return selected_data, fair_attr


def get_data_Room_Occupancy_Estimation():
    attr_string = "S1_Temp,S2_Temp,S3_Temp,S4_Temp,S1_Light,S2_Light,S3_Light,S4_Light,S1_Sound,S2_Sound," \
                  "S3_Sound,S4_Sound,S5_CO2,S5_CO2_Slope,S6_PIR,S7_PIR,Room_Occupancy_Count"
    data = pd.read_csv(
        project_path + 'Room_Occupancy_Estimation_processed.csv',
    )
    attributes = attr_string.split(',')
    # 敏感属性
    fair_attr = 'S7_PIR'  # S7_PIR
    selected_data = data[attributes]
    # 非数字型属性处理

    # 从1编号排序
    selected_data.reset_index(drop=True, inplace=True)
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_Rice():
    attr_string = 'Area Integer,Perimeter Real,Major_Axis_Length Real,Minor_Axis_Length Real,' \
                  'Eccentricity	RealConvex_Area	Integer,Extent Real,Class'
    data = pd.read_csv(
        project_path + 'Rice_processed.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # # 敏感属性
    fair_attr = 'Class'
    # # 非数字型处理
    # mapping = {'Cammeo': 0, 'Osmancik': 1}
    # selected_data[fair_attr] = selected_data[fair_attr].map(mapping)



    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_Wholesale():
    attr_string = 'Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen'
    data = pd.read_csv(
        project_path + 'Wholesale.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'Region'
    # fair_attr = 'Region' # Channel

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_student():
    attr_string = 'sex;age;Medu;Fedu'
    data = pd.read_csv(
        project_path + 'student.csv'
        , delimiter=';', quotechar='"'
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(';')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'sex'
    # 非数字型处理
    mapping = {'F': 0, 'M': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_abalone():
    data = pd.read_csv(
        project_path + 'abalone_processed.csv',
    )
    # 给定的属性字符串
    data_string = "Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings"
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = data_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'Sex'
    # # 非数字型处理
    # sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    # selected_data['Sex'] = selected_data['Sex'].map(sex_mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_parkinsons():
    attr_string = 'MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),' \
                  'MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,' \
                  'MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,' \
                  'NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE'
    data = pd.read_csv(
        project_path + 'parkinsons.data',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'status'
    # 非数字型处理
    # mapping = {'F': 0, 'M': 1}
    # selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # 随机样本数
    # num_samples = 195
    # selected_data = selected_data.sample(num_samples, random_state=None)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_vertebral():
    attr_string = 'pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,sacral_slope,pelvic_radius,' \
                  'degree_spondylolisthesis,class'

    data = pd.read_csv(
        project_path + 'vertebral.csv'
        , delimiter=' '
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'class'
    # 非数字型处理
    mapping = {'AB': 0, 'NO': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # 随机样本数
    # num_samples = 310
    # selected_data = selected_data.sample(num_samples, random_state=None)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_liver_disorder():
    attr_string = 'mcv,alkphos,sgpt,sgot,gammagt,drinks,selector'

    data = pd.read_csv(
        project_path + 'liver_disorder.data',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'selector'
    # 非数字型处理

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_heart_failure_clinical():
    attr_string = 'age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,' \
                  'serum_creatinine,platelets,serum_sodium,sex,smoking,time,DEATH_EVENT'

    data = pd.read_csv(
        project_path + 'heart_failure_clinical.csv',
    )
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'sex'

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_chronic_kidney_disease():
    attr_string = 'age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc'

    data = pd.read_csv(
        project_path + 'chronic_kidney_disease.csv',
    )
    # 替换问号为0
    data.replace('?', 0, inplace=True)
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'class'
    # 非数字型处理
    mapping = {'ckd': 0, 'notckd': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # 随机样本数
    # num_samples = 400
    # selected_data = selected_data.sample(num_samples, random_state=None)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr


def get_data_dermatology():
    attr_string = 'erythema,scaling,definite_borders,itching,koebner_phenomenon,polygonal_papules,follicular_papules,' \
                  'oral_mucosal_involvement,knee_and_elbow_involvement,scalp_involvement,family_history,melanin_incontinence,' \
                  'eosinophils_in_the_infiltrate,PNL_infiltrate,' \
                  'fibrosis_of_the_papillary_dermis,exocytosis,acanthosis,hyperkeratosis,parakeratosis,' \
                  'clubbing_of_the_rete_ridges,elongation_of_the_rete_ridges,thinning_of_the_suprapapillary_epidermis,' \
                  'spongiform_pustule,munro_microabcess,focal_hypergranulosis,disappearance_of_the_granular_layer,' \
                  'vacuolisation_and_damage_of_basal_layer,spongiosis,' \
                  'saw-tooth_appearance_of_retes,follicular_horn_plug,perifollicular_parakeratosis,' \
                  'inflammatory_monoluclear_infiltrate,band-like_infiltrate,Age'

    data = pd.read_csv(
        project_path + 'dermatology.data',
    )
    # 替换问号为0
    data.replace('?', 0, inplace=True)
    # 使用split()方法按逗号分割字符串 # 聚类属性和敏感属性
    attributes = attr_string.split(',')
    # 提取指定的属性列
    selected_data = data[attributes]

    # 敏感属性
    fair_attr = 'erythema'
    # 非数字型处理
    # mapping = {'ckd': 0, 'notckd': 1}
    # selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # 重新为数据框中的点编号，从1开始排序
    selected_data.reset_index(drop=True, inplace=True)
    # 重新为数据框中的点编号，从2开始排序
    selected_data.index = selected_data.index + 1
    return selected_data, fair_attr
