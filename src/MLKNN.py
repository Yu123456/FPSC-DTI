# coding:utf-8

# 本程序编写于 2018 年 11 月 12 日
# 按照
# Jian-Yu Shi, et al. Predicting drug-target interaction for new drugs using enchanced
# similarity measures and super-target clustering. Methods. 2015.
# 概率估计，实现 MLKNN

#########################################################################################
'''
关于 MLKNN class 设计：
1、设计成 drug-target 的预测时，比较特殊，需要输入一个相似度矩阵 trainSim, 及相互作用矩阵
   trainY, 为此，进行如下对应：
   如果 trainSim 用 drug 相似度矩阵，则 trainY -> drug * target 形式的作用矩阵
   如果 trainSim 用 target 相似度矩阵，则 trainY -> target * drug 形式的作用矩阵
   即输入 drug (target) 相似度矩阵，估计 target t_i (drug d_i) 与 drug (target) 的作用
   概率，因为
   Pr[a(x,j)=1]\sim [1 + \sum_{i=1}^m A(i,j)] / (m+2)
   表示 drug d_x 与 target t_j 的概率
2、预测方面，仅是针对训练集 trainY ， 预测 所有 作用关系的概率值，因此，预测函数中无需输入
   实参
'''
##########################################################################################

import numpy as np

# MLKNN class
class MLKNN:
    '''
    类属性：
    训练集相似度矩阵 trainSim
    训练集作用矩阵   trainY
    相互作用先验概率 Prior
    无相互作用先验概率 PriorN  = 1 - Prior
    相互作用条件概率 ConditionPr
    无相互作用条件概率 ConditionPrN
    KNN   K  默认值为 3
    概率估计调整参数 smooth   默认值为 1

    类成员：
    训练函数 fit
    预测函数 predict
    '''

    # 初始化函数
    def __init__(self,K=3,smooth=1):
        self.K = K
        self.smooth = smooth
        self.trainSim = None
        self.trainY = None
        self.Prior = None
        self.PriorN = None
        self.ConditionPr = None
        self.ConditionPrN = None

    # 训练函数
    def fit(self,trainSim,trainY):

        self.trainSim = np.copy(trainSim)   # 矩阵
        self.trainY = np.copy(trainY)       # 矩阵

        rows,cols = self.trainY.shape       # 行数、列数
        # 计算先验概率
        # 由于 trainY 矩阵中元素就是 0 或者 1，可以直接相加，统计作用关系数量
        # 按照 Eq.(2), 按列求和
        tempCi = np.sum(self.trainY,axis=0)   # 生成 cols 列, 形状为 (cols, )
        Prior = (1.0 * self.smooth + tempCi) / (self.smooth * 2.0 + rows)
        PriorN = 1.0 - Prior

        self.Prior = Prior
        self.PriorN = PriorN

        # 计算条件概率
        neighbor = np.argsort(-self.trainSim,axis=1)  # 按行递减排序，返回索引
        tempCi = np.zeros((cols,self.K + 1),dtype=float)   # trainY 每一列都有 K+1 个概率值，kNN 有 K +1 个不同取值
        tempCiN = np.zeros((cols,self.K + 1),dtype=float)
        for i in range(rows):
            # trainSim 中第 i 个 drug or target
            neighbor_index = neighbor[i,range(self.K)]   # trainSim_i 的 K 近邻索引
            neighbor_label = self.trainY[neighbor_index,:]
            # temp 形状为 (cols, )
            temp = np.sum(neighbor_label,axis=0)    # 按列求和，trainY 每一列与 trainSim_i 的 K 近邻作用关系个数

            # trainY 第 i 行作用关系
            index = self.trainY[i,:]
            for j,val in enumerate(index):
                index_col = int(temp[j])  # trainY 中第 j 列与 trainSim_i 的 K 近邻中作用关系个数
                if val == 1:
                    tempCi[j,index_col] = tempCi[j,index_col] + 1
                else:
                    tempCiN[j,index_col] = tempCiN[j,index_col] + 1

        temp1 = np.sum(tempCi,axis=1)  # 按行求和，即所有近邻值求和
        temp2 = np.sum(tempCiN,axis=1)
        ConditionPr = np.zeros((cols,self.K + 1),dtype=float)
        ConditionPrN = np.zeros((cols,self.K + 1),dtype=float)
        for i in range(cols):
            ConditionPr[i,:] = (1.0 * self.smooth + tempCi[i,:]) / (1.0 * self.smooth * (self.K +1) + temp1[i])
            ConditionPrN[i,:] = (1.0 * self.smooth + tempCiN[i,:]) / (1.0 * self.smooth * (self.K + 1) + temp2[i])

        self.ConditionPr = ConditionPr
        self.ConditionPrN = ConditionPrN

    # 预测函数
    def predict(self):
        rows,cols = self.trainY.shape   # 预测每一个 row_i 与所有 cols 的概率值
        neighbor = np.argsort(-self.trainSim, axis=1)  # 按行递减排序，返回索引

        # 输出概率值
        Outputs = np.zeros((rows,cols),dtype=float)
        for i in range(rows):
            neighbor_index = neighbor[i,range(self.K)]  # trainSim_i 的 K 近邻索引
            neighbor_label = self.trainY[neighbor_index, :]
            # temp 形状为 (cols, )
            temp = np.sum(neighbor_label, axis=0)  # 按列求和，trainY 每一列与 trainSim_i 的 K 近邻作用关系个数
            temp = [int(x) for x in temp]     # 取整
            ProbIn = np.dot(self.Prior,self.ConditionPr[:,temp])
            ProbOut = np.dot(self.PriorN,self.ConditionPrN[:,temp])
            P2 = ProbIn + ProbOut

            # 一些特殊情形处理，如 P2 中元素为 0
            isZeros_index = np.where(P2 < 0.0001)
            isZeros_index = isZeros_index[0]
            isNotZeros_index = set(range(cols)) - set(isZeros_index)
            isNotZeros_index = list(isNotZeros_index)  # 转成 list
            if(len(isZeros_index) == cols):
                # 所有元素均为 0
                Outputs[i,:] = self.Prior
            elif(len(isNotZeros_index) == cols):
                # 所有元素均非 0
                Outputs[i,:] = ProbIn / P2
            else:
                # 既有 0 ，也有非 0
                # 0 位置用先验概率代替
                Outputs[i,isZeros_index] = self.Prior[isZeros_index]
                # 非 0 位置用贝叶斯推理概率
                Outputs[i,isNotZeros_index] = ProbIn[isNotZeros_index] / P2[isNotZeros_index]

        return Outputs


# Super Cluster
def SuperClusterDTI(label,Y):
    '''

    :param label: 一维标签向量
    :param Y:    作用关系矩阵，行数对应着 label 向量长度
    :return:
    '''
    Ys = np.copy(Y)
    ulabel = np.unique(label)
    for lab in ulabel:
        index = np.where(label == lab)
        index = index[0]    # 从 tuple 中取出索引
        super_label = np.max(Y[index,:],axis=0)    # 按列求最大值，当某列有 1 时，该列值为 1，否则为 0
        Ys[index,:] = super_label

    return Ys


# Super Cluster Score
# Score = SuperDrugScore * SuperTargetScore * DrugScore * TargetScore
# SuperDrugScore 对 drug 聚类成 SuperDrug 预测概率值
# SuperTargetScore 对 target 聚类成 SuperTarget 预测概率值
# DrugScore 从 drug 近邻直接对原始的 interaction 预测概率值
# TargetScore 从 target 近邻直接对原始的 interaction 预测概率值
def SuperClusterScore(sim_drug,sim_target,admat,label_drug,label_target,K=3,smooth=1.0):
    '''

    :param sim_drug:  drug 相似度矩阵
    :param sim_target:  target 相似度矩阵
    :param admat:  drug-target interaction 矩阵，维度为 drug * target
    :param K:    kNN 的 K 值，默认为 3
    :param smooth:  MLKNN 的概率估计调整参数，默认为 1.0
    :param label_drug: drug 的类别标签
    :param label_target: target 的类别标签
    :return: 返回与 admat 维度一致的预测概率值
    '''

    # copy
    drug = np.copy(sim_drug)
    target = np.copy(sim_target)
    dt_admat = np.array(np.copy(admat),dtype=int)  # drug * target 维度
    label_d = np.copy(label_drug)
    label_t = np.copy(label_target)

    # DrugScore
    clf = MLKNN()
    clf.fit(trainSim=drug,trainY=dt_admat)
    DrugScore = clf.predict()   # drug * target

    # TargetScore
    clf = MLKNN()
    clf.fit(trainSim=target,trainY=dt_admat.T)
    TargetScore = clf.predict()     # target * drug
    TargetScore = TargetScore.T     # drug * target

    # 勘误前
    # SuperDrugScore
    # new_admat = SuperClusterDTI(label=label_d,Y=dt_admat)   # drug * target
    # clf = MLKNN()
    # clf.fit(trainSim=drug,trainY=new_admat)
    # SuperDrugScore = clf.predict()    #    drug * target

    # 勘误后  ------ 2019-02-17
    # SuperDrugScore
    new_admat = SuperClusterDTI(label=label_d, Y=dt_admat)  # drug * target
    clf = MLKNN()
    clf.fit(trainSim=target, trainY=new_admat.T)      # target * drug
    SuperDrugScore = clf.predict()  # target * drug
    SuperDrugScore = SuperDrugScore.T  # drug * target

    # 勘误前
    # SuperTargetScore
    # new_admat = SuperClusterDTI(label=label_t,Y=dt_admat.T)  # target * drug
    # clf = MLKNN()
    # clf.fit(trainSim=target,trainY=new_admat)
    # SuperTargetScore = clf.predict()     # target * drug
    # SuperTargetScore = SuperTargetScore.T   # drug * target

    # 勘误后  ------- 2019-02-17
    # SuperTargetScore
    new_admat = SuperClusterDTI(label=label_t, Y=dt_admat.T)  # target * drug
    clf = MLKNN()
    clf.fit(trainSim=drug, trainY=new_admat.T)     # drug * target
    SuperTargetScore = clf.predict()  # drug * target

    # max model
    dt_score = np.maximum(DrugScore,TargetScore)
    dt_super_score = np.maximum(SuperDrugScore,SuperTargetScore)

    return dt_score * dt_super_score

