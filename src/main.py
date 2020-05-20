# coding:utf-8
# 2020-5

# 将 SuperTarget 模型与 FPRM 模型结合预测

# 1. 在多维尺度化（MDS）特征基础上，应用 FPRM 方法预测作用关系得分 score1
# 2. 在相似度矩阵、作用矩阵基础上，应用 SuperCluster 方法预测作用关系得分 score2
# 3. 将两个模型的预测得分合并 score = score1 * score2  (乘积方式)

# 用 MPR 进行评价，仅对正例进行 10-folds cv
# 需要对待测集进行置 0 处理
#
# 预测所有 unknown-interaction ，所有正例参与训练，与 checked interaction
# 进行 top N 评价

# FPRM 参数如下：
# re_lambda

# SuperCluster 参数如下：
# K = 3  kNN 值
# smooth = 1.0    光滑参数

############################################################################################

import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from crossValidation import CrossValidationF as CVF
from topN import topN
from FPRMClass import FPRMmembership
from MLKNN import SuperClusterScore


#############################################################################################
# 本地机器数据地址
dataPath = 'C:/Users/YDH/Documents/Python2/DrugTargetPrediction/Data/'
cluster_path = 'C:/Users/YDH/Documents/Python2/DrugTargetPrediction/Data/INCK-Cluster/'

# C44 服务器
# dataPath = '/media/data/users/phd/2015/yudonghua/python2/FPRModel/Data/'

# 数据集名称
# 多维尺度化数据 .mat
mdsName = ['Enzyme','GPCR','IonChannel','NuclearRecept']
# 原始数据集，相似度矩阵，作用矩阵
originalNameAdmat = ['AdmatEnzyme','AdmatGPCR','AdmatIonChannel','AdmatNuclearRecept']
originalNameT = ['TsimmatEnzyme','TsimmatGPCR','TsimmatIonChannel','TsimmatNuclearRecept']
originalNameC = ['CsimmatEnzyme','CsimmatGPCR','CsimmatIonChannel','CsimmatNuclearRecept']
# checked interaction data   .csv
checkedName = ['EnzymeResultCheck.csv','GPCRResultCheck.csv','ICResultCheck.csv','NRResultCheck.csv']

# 聚类结果数据
clusterD_name = ['EnzymeDrugClusterINCK','GPCRDrugClusterINCK','IonChannelDrugClusterINCK','NuclearReceptDrugClusterINCK']
clusterT_name = ['EnzymeTargetClusterINCK','GPCRTargetClusterINCK','IonChannelTargetClusterINCK','NuclearReceptTargetClusterINCK']

# 数据保存名称
# 预测结果 score 排序并合并 checked interaction 结果
# savedCheckedName = []

# MPR 排序后，参数输出保存
sortMPR_name = ['sortMPREnzyme.csv','sortMPRGPCR.csv','sortMPRIC.csv','sortMPRNR.csv']

# 当前数据集
# currData = 0     # Enzyme data
# currData = 1     # GPCR data
currData = 2     # IC data
# currData = 3       # NR data

print('数据集：%s' %mdsName[currData])

# 读取 MDS 数据
mdsData = sio.loadmat(dataPath+'mds-mat/'+mdsName[currData])
# drug 特征， drug 数 * 特征维数, 序号对应 interaction 列顺序
drugAttribute = np.array(mdsData['drugAttribute'],dtype=float)
# target 特征， target 数 * 特征维数, 序号对应 interaction 行顺序
targetAttribute = np.array(mdsData['targetAttribute'],dtype=float)

# 对 drug, target 特征进行归一化
clfPreprocess = MinMaxScaler()
drugAttribute = clfPreprocess.fit_transform(drugAttribute)
targetAttribute = clfPreprocess.fit_transform(targetAttribute)

# 读取 original interaction matrix
originalData = sio.loadmat(dataPath+'original-mat/'+originalNameAdmat[currData])
interactionMat = np.array(originalData['matrix'],dtype=int)  # target * drug
drugID = originalData['drugID']  # drug ID, 对应着 interaction matrix 列
targetID = originalData['targetID']  # target ID, 对应着 interaction matrix 行
dID = [x[0] for x in drugID[:,0]]
tID = [x[0] for x in targetID[:,0]]

# 读取 checked interaction .csv
df_checked = pd.read_csv(dataPath+'DT-Checked-4-csv/'+checkedName[currData])
df_checkedOnly = df_checked[['Drug ID','Target ID','Checked']]

# 读取聚类结果数据
clusterD_mat = sio.loadmat(cluster_path+clusterD_name[currData])
label_drug = clusterD_mat['cl']
label_drug = np.reshape(label_drug,(label_drug.shape[0],))

clusterT_mat = sio.loadmat(cluster_path+clusterT_name[currData])
label_target = clusterT_mat['cl']
label_target = np.reshape(label_target,(label_target.shape[0],))

# 读取相似度矩阵
simD_mat = sio.loadmat(dataPath+'original-mat/'+originalNameC[currData])
simD = simD_mat['matrix']
simD = (simD + simD.T) / 2.0     # 对称化

simT_mat = sio.loadmat(dataPath+'original-mat/'+originalNameT[currData])
simT = simT_mat['matrix']
simT = (simT + simT.T) / 2.0     # 对称化


# 重复次数，交叉验证次数
numSplit = 5
folds = 10

# top 中已经验证的个数
tops = [10,20,30,50,100,200,300,400,500,600,700,800,900,1000]


# 交叉验证 MPR 值
re_lambda = 43.198
mpr_mean, mpr_std = CVF(interactionMatrix=interactionMat,drugAttribute=drugAttribute,
                        targetAttribute=targetAttribute,re_lambda=re_lambda,
                        sim_drug=simD, sim_target=simT, label_drug=label_drug,
                        label_target=label_target,numSplit=numSplit,kfolds=folds,
                        K=3,smooth=1.0)
print('lambda = %.4f , MPR = %.4f +/- %.4f' %(re_lambda,mpr_mean,mpr_std))

# 计算 top N list
# 预测所有 unknown-interaction

# FPRM
clf = FPRMmembership(re_lambda=re_lambda)
clf.fit(X=targetAttribute,Y=drugAttribute,Delta=interactionMat)
score1 = clf.predict_score_matrix(X=targetAttribute,Y=drugAttribute,label=1)
# SuperCluster
# score2 的形式与 admat 一致，即 drug * target
score2 = SuperClusterScore(sim_drug=simD,sim_target=simT,admat=interactionMat.T,
                          label_drug=label_drug,label_target=label_target)
# 预测得分合并
score = score1 * score2.T

# 将所有 unknown-interaction 预测值抽取出来，组成一个 DataFrame
# rowIndex --> target, colIndex -- drug
rowIndex, colIndex = np.where(interactionMat == 0)
df_score = pd.DataFrame({'score':score[rowIndex,colIndex],'rowIndex':rowIndex,'colIndex':colIndex})
# 2. 计算 top N, 返回 list
tops_result = topN(df_score=df_score,list_dID=dID,list_tID=tID,list_tops=tops,df_checked=df_checkedOnly)
print tops_result
