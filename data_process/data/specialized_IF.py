# �������ݼ��ĵ�������취
import networkx as nx
import numpy as np
import pandas as pd
import random
from io import StringIO
from sklearn.metrics import euclidean_distances

pd.options.mode.chained_assignment = None  # default='warn '

project_path = 'D:\\env\\py_all\\cluster\\FDPC-main\\FDPC-main\\data_process\\data\\fairness_data\\'
project_path2 = 'D:\\env\\py_all\\cluster\\FDPC-main\\FDPC-main\\data_process\\data\\dataset\\'

seed = 1
def create_graph_from_data(data, threshold):
    G = nx.Graph()
    distances = euclidean_distances(data)
    num_points = data.shape[0]

    # ���������ת��Ϊһά���飬����������ٷ�λ���ľ�����ֵ
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
    attr_string = 'RI,Na,Mg,Al,Si,K,Ca,Ba,Fe'

    data = pd.read_csv(
        project_path2 + 'glass.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ���������

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_seeds():
    attr_string = 'area,perimeter,compactness,length of kernel,width of kernel,asymmetry coefficient,length of kernel groove,class'

    data = pd.read_csv(
        project_path2 + 'seeds.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ���������

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_wdbc():
    attr_string = 'radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,' \
                  'concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_SE,texture_SE,' \
                  'perimeter_SE,area_SE,moothness_SE,compactness_SE,concavity_SE,concave points_SE,symmetry_SE,fractal_dimension_SE,' \
                  'radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,' \
                  'concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst'

    data = pd.read_csv(
        project_path2 + 'wdbc.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_wine():
    attr_string = 'Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,' \
                  'Nonflavanoid phenols,Proanthocyanins,' \
                  'Color intensity,Hue,OD280/OD315 of diluted wines,Proline'

    data = pd.read_csv(
        project_path2 + 'wine.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ���������

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_iris():
    # ��������
    data = pd.read_csv(
        project_path2 + 'iris.csv')
    # �������Ժ���������
    attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # ��������
    # fair_attr = 'class'
    # ��ȡָ����������
    selected_data = data[attributes]
    # # �����������Դ���
    # sex_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    # selected_data['class'] = selected_data['class'].map(sex_mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : ���Ϊ1�ĵ��'class'����ֵ
    return selected_data


def get_data_adult():
    # ��������
    data = pd.read_csv(
        project_path + 'adult.data'  # adult.data
    )
    # print(data.columns)
    # �������Ժ���������
    atrr_string = "age,education-num,hours-per-week,sex"
    attributes = atrr_string.split(',')
    # print(attributes)
    # ��ȡָ����������
    selected_data = data[attributes]
    # �����������Դ���
    num_samples = (int)(len(selected_data) * 0.2)
    selected_data = selected_data.sample(num_samples, random_state=seed)
    sex_attr = 'sex'
    sex_attr_mapping = {' Female': 0, ' Male': 1}
    selected_data[sex_attr] = selected_data[sex_attr].map(sex_attr_mapping).fillna(-1).astype(int)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_bank():
    # ��������
    data = pd.read_csv(
        project_path + 'bank.csv', # _processed
        delimiter=';', quotechar='"'
    )
    # �������Ժ���������
    attr_str = 'age,balance,duration,balance,marital'
    attributes = attr_str.split(',')  # 'education'
    # ��������
    fair_attr = 'marital'
    # ��ȡָ����������
    selected_data = data[attributes]
    # �����������Դ���
    marital_mapping = {'single': 0, 'married': 1, 'divorced': 2}
    selected_data[fair_attr] = selected_data[fair_attr].map(marital_mapping).fillna(-1).astype(int)
    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_obesity():
    # ��������
    data = pd.read_csv(
        project_path + 'obesity.csv')
    # �������Ժ���������
    attributes = ['Age', 'Height', 'Weight', 'Gender']

    # ��ȡָ����������
    selected_data = data[attributes]
    # �����������Դ���
    sex_mapping = {'Female': 0, 'Male': 1}
    selected_data['Gender'] = selected_data['Gender'].map(sex_mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : ���Ϊ1�ĵ��'class'����ֵ
    return selected_data


def get_data_census1990():
    # ��������
    data = pd.read_csv(
        project_path + 'census1990.csv')
    # �������Ժ���������
    attributes = ['dAge', 'dAncstry1', 'dAncstry2', 'iAvail', 'iCitizen', 'iClass', 'dDepart', 'iDisabl1',
                  'iDisabl2', 'iEnglish', 'iFeb55', 'iFertil', 'dHispanic', 'dHour89', 'dHours', 'iImmigr',
                  'dIncome1', 'dIncome2', 'dIncome3', 'dIncome4', 'dIncome5', 'dIncome6', 'dIncome7',
                  'dIncome8', 'dIndustry', 'iKorean', 'iLang1', 'iLooking', 'iMarital', 'iMay75880', 'iMeans',
                  'iMilitary', 'iMobility', 'iMobillim', 'dOccup', 'iOthrserv', 'iPerscare', 'dPOB',
                  'dPoverty', 'dPwgt1', 'iRagechld', 'dRearning', 'iRelat1', 'iRelat2', 'iRemplpar',
                  'iRiders', 'iRlabor', 'iRownchld', 'dRpincome', 'iRPOB', 'iRrelchld', 'iRspouse',
                  'iRvetserv', 'iSchool', 'iSept80', 'iSex', 'iSubfam1', 'iSubfam2', 'iTmpabsnt',
                  'dTravtime', 'iVietnam', 'dWeek89', 'iWork89', 'iWorklwk', 'iWWII', 'iYearsch',
                  'iYearwrk', 'dYrsserv']

    # ��ȡָ����������
    selected_data = data[attributes]
    num_samples = (int)(len(selected_data) * 0.012)
    selected_data = selected_data.sample(num_samples, random_state=seed)
    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_creditcard():
    # ��������
    data = pd.read_csv(
         project_path + 'creditcard.csv')
    # �����������ַ���
    data_string = "LIMIT_BAL,EDUCATION,SEX,BILL_AMT1,BILL_AMT2," \
                  "BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6"

    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = data_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]
    # ���������
    num_samples = (int)(len(selected_data) * 0.2)
    selected_data = selected_data.sample(num_samples, random_state=seed)
    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1

    return selected_data


def get_data_drug_consumption():
    # ��������
    attr_string = "ID,Age,Gender,Education,Country,Ethnicity,Nscore,Escore,Oscore,Ascore,Cscore,Impulsive,SS"
    data = pd.read_csv(
        project_path + 'drug_consumption.data')

    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')

    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : ���Ϊ1�ĵ��'class'����ֵ
    return selected_data


# ʧ��
def get_data_drug():
    # ��������
    data = pd.read_csv(
        project_path + 'drug.csv',
    )

    # �����������ַ���
    data_string = "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28," \
                  "29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47," \
                  "85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102," \
                  "103,104,105,106,107,108,109,110,112,113,114,115,116,117,118,119,120,121," \
                  "122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140," \
                  "141,142,143,144,145,146,147,148,149,150,151,152,153,154,Ethnicity,Gender,HasTie"
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = data_string.split(',')

    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_dabestic():
    # ��������
    data = pd.read_csv(
        project_path + 'dabestic.csv'
        , delimiter='\t')
    # data �ǰ����Ʊ���ָ������ݵ��ַ���
    # �������Ժ���������
    attributes = ['AGE', 'SEX', 'BMI', 'S1', 'S2', 'S3', 'BP', 'S6']

    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : ���Ϊ1�ĵ��'class'����ֵ
    return selected_data


def get_data_hcvdat0():
    data = pd.read_csv(
        project_path + 'hcvdat0.csv',
    )
    # print(data.columns)
    # �������Ժ���������
    atrr_string = "Sex,AST,BIL,CHE,CREA,GGT"
    attributes = atrr_string.split(',')
    # print(attributes)
    # ��������
    sex_attr = 'Sex'
    # ��ȡָ����������
    selected_data = data[attributes]
    # �����������Դ���
    sex_mapping = {'m': 0, 'f': 1}
    selected_data.loc[:, sex_attr] = selected_data[sex_attr].map(sex_mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : ���Ϊ1�ĵ��'class'����ֵ
    return selected_data


def get_data_athlete():
    # ��������
    data = pd.read_csv(
        project_path + 'athlete.csv'
        , quotechar='"'
    )
    # �滻���е�NaNֵΪ-1
    data = data.fillna(-1)
    # print(data.columns)
    # ��������
    atrr_string = "Sex,Age,Height,Weight,Year,Season"
    attributes = atrr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    num_samples = (int)(len(selected_data) * 0.023)
    selected_data = selected_data.sample(num_samples, random_state=seed)

    sex_mapping = {'M': 0, 'F': 1}
    season_mapping = {'Summer': 0, 'Winter': 1}
    selected_data.loc[:, 'Season'] = selected_data['Season'].map(season_mapping)
    selected_data.loc[:, 'Sex'] = selected_data['Sex'].map(sex_mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    # eg: data[1]['class'] : ���Ϊ1�ĵ��'class'����ֵ
    return selected_data


def get_data_Room_Occupancy_Estimation():
    attr_string = "S1_Temp,S2_Temp,S3_Temp,S4_Temp,S1_Light,S2_Light,S3_Light,S4_Light,S1_Sound,S2_Sound," \
                  "S3_Sound,S4_Sound,S5_CO2,S5_CO2_Slope,S6_PIR,S7_PIR,Room_Occupancy_Count"
    data = pd.read_csv(
        project_path + 'Room_Occupancy_Estimation.csv',
    )
    attributes = attr_string.split(',')
    selected_data = data[attributes]
    # �����������Դ���
    num_samples = (int)(len(selected_data) * 0.6)

    selected_data = selected_data.sample(num_samples, random_state=seed)

    # ��1�������
    selected_data.reset_index(drop=True, inplace=True)
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_Rice():
    attr_string = 'Area Integer,Perimeter Real,Major_Axis_Length Real,Minor_Axis_Length Real,' \
                  'Eccentricity	RealConvex_Area	Integer,Extent Real'
    data = pd.read_csv(
        project_path + 'Rice.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_Wholesale():
    attr_string = 'Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen'
    data = pd.read_csv(
        project_path + 'Wholesale.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_student():
    attr_string = 'sex;age;Medu;Fedu'
    data = pd.read_csv(
        project_path + 'student.csv'
        , delimiter=';', quotechar='"'
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(';')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ��������
    fair_attr = 'sex'
    # �������ʹ���
    mapping = {'F': 0, 'M': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_abalone():
    data = pd.read_csv(
        project_path + 'abalone_processed.csv',
    )
    # �����������ַ���
    data_string = "Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings"
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = data_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # # ��������
    # fair_attr = 'Sex'
    # # # �������ʹ���
    #sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    #selected_data['Sex'] = selected_data['Sex'].map(sex_mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_parkinsons():
    attr_string = 'MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),' \
                  'MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,' \
                  'MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,' \
                  'NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE'
    data = pd.read_csv(
        project_path + 'parkinsons.data',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ��������
    fair_attr = 'status'
    # �������ʹ���
    mapping = {'F': 0, 'M': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # ���������
    # num_samples = 195
    # selected_data = selected_data.sample(num_samples, random_state=None)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_vertebral():
    attr_string = 'pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,sacral_slope,pelvic_radius,' \
                  'degree_spondylolisthesis'

    data = pd.read_csv(
        project_path + 'vertebral.csv'
        , delimiter=' '
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_liver_disorder():
    attr_string = 'mcv,alkphos,sgpt,sgot,gammagt,drinks'

    data = pd.read_csv(
        project_path + 'liver_disorder.data',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]


    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_heart_failure_clinical():
    attr_string = 'age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,' \
                  'serum_creatinine,platelets,serum_sodium,sex,smoking,time,DEATH_EVENT'

    data = pd.read_csv(
        project_path + 'heart_failure_clinical.csv',
    )
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


def get_data_chronic_kidney_disease():
    attr_string = 'age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc'

    data = pd.read_csv(
        project_path + 'chronic_kidney_disease.csv',
    )
    # �滻�ʺ�Ϊ0
    data.replace('?', 0, inplace=True)
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ��������
    fair_attr = 'class'
    # �������ʹ���
    mapping = {'ckd': 0, 'notckd': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)


    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data


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
    # �滻�ʺ�Ϊ0
    data.replace('?', 0, inplace=True)
    # ʹ��split()���������ŷָ��ַ��� # �������Ժ���������
    attributes = attr_string.split(',')
    # ��ȡָ����������
    selected_data = data[attributes]

    # ��������
    fair_attr = 'erythema'
    # �������ʹ���
    mapping = {'ckd': 0, 'notckd': 1}
    selected_data[fair_attr] = selected_data[fair_attr].map(mapping)

    # ����Ϊ���ݿ��еĵ��ţ���1��ʼ����
    selected_data.reset_index(drop=True, inplace=True)
    # ����Ϊ���ݿ��еĵ��ţ���2��ʼ����
    selected_data.index = selected_data.index + 1
    return selected_data
