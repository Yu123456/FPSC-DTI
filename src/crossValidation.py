# coding:utf-8

# Cross Validation and Evaluate MPR


'''
本代码是为 drug-target interaction 中作用矩阵做交叉验证设计，
只对正例进行交叉。
参考 Ming Hao, et al. Open-source chemogenomic data-driven algorithms for
predicting drug-target interactions. 2018 .
中的提供的 R 代码
主要思想：
1、只对正例进行交叉验证
2、测试集中每一行、每一列非全 0

2019-01-04
doCVPositiveOnly3 只是对正例进行交叉验证，无法用于计算 AUC, AUPR 等值，因此
重新设计一个交叉函数，以进行 AUPR, AUC 的计算，此时需要对正负例都进行交叉，
该函数命名为 doCrossValidation, 以与文献提供的 R 语言代码相一致
'''





# 作为一个交叉验证函数，需要返回测试集、训练集在作用矩阵中的索引

import numpy as np
from sklearn.model_selection import KFold  # k-folds CV
import pandas as pd
from random import sample
import sys

from meanPercentileRank import evalMPR
from FPRMClass import FPRMmembership
from MLKNN import SuperClusterScore


def get_k_folds(tsize,kfolds=10):
    '''

    :param tsize:  需要交叉验证的样本总数
    :param kfolds:
    :return:
    '''

    indices = np.array(range(tsize))
    np.random.shuffle(indices)
    testFolds = []
    trainFolds = []
    skf = KFold(n_splits=kfolds)
    for trIndex, teIndex in skf.split(indices):
        Xtr,Xte = indices[trIndex], indices[teIndex]
        testFolds.append(Xte.tolist())
        trainFolds.append(Xtr.tolist())

    return testFolds,trainFolds


def doCVPositiveOnly3(inMat,kfold=10,numSplit=5):
    '''

    :param inMat: 作用矩阵，元素为 {0,1},  target * drug
    :param kfold:
    :param numSplit:
    :return:
    '''

    tempMat = np.array(inMat)
    # print tempMat
    rowInds,colInds = np.where(tempMat == 1)    # rowInds : target,  colInds: drug
    # print rowInds
    # print colInds
    # exit()
    # print type(rowInds)
    df_ONEs = pd.DataFrame(data={'rowIndex':rowInds,'colIndex':colInds})
    df_ONEs['value'] = 1     # 增加一列
    # print df_ONEs
    # exit()
    # dff = df.groupby(by='rowIndex')
    #
    # dffv = dff['value']
    # # 分组后ID（行名称）
    # print dffv.size().index
    # print dffv.size().index[0] == 0
    # # 分组后ID（行名称）对应统计值
    # print dffv.size().values
    # print dffv.size().index[dffv.size().values > 1]

    df_row_size = df_ONEs.groupby(by='rowIndex')['value'].size()
    ind = df_row_size.index[df_row_size.values > 2]
    df = df_ONEs[df_ONEs['rowIndex'].isin(ind)]
    df_col_size = df.groupby(by='colIndex')['value'].size()
    ind = df_col_size.index[df_col_size.values > 2]
    df = df[df['colIndex'].isin(ind)]

    # print df
    # print df.drop([0,5,24])     # 按 index 值删除
    # exit()

    numTriOnes = df.shape[0]
    savedFolds = []

    for i in range(numSplit):
        # 第 i 次重复
        testFolds = []
        isThisFolds = True
        while isThisFolds:
            testFolds, trainFolds = get_k_folds(tsize=numTriOnes,kfolds=kfold)
            for j in range(kfold):
                df_Index = testFolds[j]
                currMat = np.copy(tempMat)
                rIndex = df.iloc[df_Index]['rowIndex'].values
                cIndex = df.iloc[df_Index]['colIndex'].values

                currMat[rIndex,cIndex] = 0
                rs = np.sum(currMat,axis=1)  # 行求和，第二维求和
                cs = np.sum(currMat,axis=0)  # 列求和，第一维求和
                if 0 in rs or 0 in cs:
                    isThisFolds = True
                    # jump out for-loop and perform while() function
                    break
                else:
                    # quit while() function
                    isThisFolds = False

        # 退出 while 循环，说明 testFolds 符合要求
        list_i = []   # 保存第 i 个 Split 的 k-folds cv 数据
        for j in range(kfold):
            dict_j = {}    # 以字典形式保存第 j 折的数据
            df_Index = testFolds[j]
            # 测试集索引
            testIndexRow = df.iloc[df_Index]['rowIndex'].values
            testIndexCol = df.iloc[df_Index]['colIndex'].values

            # 测试集，保存测试集索引
            testSet = df.iloc[df_Index][['colIndex','rowIndex']]  # DataFrame 格式


            # known information for drug-target matrix
            # 可以从 df_ONEs 中删除 df_ONES.iloc[df_Index]
            # 用 df_ONEs.drop 时，需要输入 index
            # print df_ONEs
            # print df
            # print df.iloc[df_Index]
            # print df.iloc[df_Index].index
            # print df_ONEs.drop(df.iloc[df_Index].index)
            # print df_ONEs

            df_drop = df_ONEs.drop(df.iloc[df_Index].index)
            knownDrugIndex = df_drop['colIndex'].values
            knownTargetIndex = df_drop['rowIndex'].values
            # print 'df_drop[colIndex].values'
            # print df_drop['colIndex'].values

            tmp = np.copy(tempMat)
            rIndex = df.iloc[df_Index]['rowIndex'].values
            cIndex = df.iloc[df_Index]['colIndex'].values
            tmp[rIndex,cIndex] = 0

            # print type(testSet)
            # print type(testIndexRow)
            # print type(testIndexCol)
            # print type(knownDrugIndex)
            # print type(knownTargetIndex)
            # print type(tmp)

            dict_j['testSet'] = testSet   # DataFrame
            dict_j['testIndexRow'] = testIndexRow  # numpy.ndarray
            dict_j['testIndexCol'] = testIndexCol  # numpy.ndarray
            dict_j['knownDrugIndex'] = knownDrugIndex  # numpy.ndarray
            dict_j['knownTargetIndex'] = knownTargetIndex  # numpy.ndarray
            dict_j['foldMat'] = tmp   # numpy.ndarray

            list_i.append(dict_j)  # 将第 j 折数据保存到 list_i 中

        savedFolds.append(list_i)

    return savedFolds

# cross validation funciton
def CrossValidationF(interactionMatrix,drugAttribute,targetAttribute,re_lambda,
                     sim_drug, sim_target,label_drug,label_target,numSplit=3,kfolds=10,
                     K=3, smooth=1.0):
    '''

    :param interactionMatrix:  drug-target interaction matrix, target * drug
    :param drugAttribute  drug 特征， drug 数 * 特征维数
    :param targetAttribute target 特征，target 数 * 特征维数
    :param re_lambda   FPRM 正则化参数
    :param numSplit: 重复次数
    :param kfolds:   交叉次数
    :param sim_drug:  drug 相似度矩阵
    :param sim_target:  target 相似度矩阵
    :param K:    kNN 的 K 值，默认为 3
    :param smooth:  MLKNN 的概率估计调整参数，默认为 1.0
    :param label_drug: drug 的类别标签
    :param label_target: target 的类别标签
    :return:
    '''

    inMat = np.copy(interactionMatrix)
    drug_attr = np.copy(drugAttribute)
    target_attr = np.copy(targetAttribute)

    # 交叉验证
    savedFolds = doCVPositiveOnly3(inMat=inMat,kfold=kfolds,numSplit=numSplit)
    # 在 savedFolds 中的作用关系矩阵 foldMat 与输入关系矩阵 inMat 形式上一致
    # 由于在这里 inMat 输入时是 target * drug 矩阵，因此，foldMat 也是 target * drug 矩阵

    mpr_list = []
    for i,spliting in enumerate(savedFolds):
        for j,folding in enumerate(spliting):
            # 准备数据集， train, test
            # 值得注意的是，MPR 评价需要 drug_i 与所有 target 的预测值
            # 因此，对于一个训练好的模型，需要预测整个 interaction matirx 中的值

            # train data
            # 交叉验证中已经返回了将测试集置 0 的作用矩阵，直接读取即可
            delta = folding['foldMat']       # target * drug 作用矩阵

            # FPRM 模型的预测得分 score1
            clf = FPRMmembership(re_lambda=re_lambda)
            clf.fit(X=target_attr,Y=drug_attr,Delta=delta)
            score1 = clf.predict_score_matrix(X=target_attr,Y=drug_attr,label=1) # target * drug 得分矩阵



            # SuperCluster 模型的预测得分 score2
            # Score2 的形式与 admat 一致，即此处需要输入 drug * target
            score2 = SuperClusterScore(sim_drug=sim_drug, sim_target=sim_target, admat=delta.T,
                                      label_drug=label_drug,label_target=label_target)

            # FPRM 与 SuperCluster 预测得分合并
            score = score1 * score2.T    # 乘积方式

            # MPR 值
            # 注意：folding['testSet'][['rowIndex','colIndex']].values 索引计数从 0 开始的，
            # 而 evalMPR 函数内默认 testSet 形参从 1 开始计数，
            # 故需要在从 0 开始计数的索引上加 1
            mpr = evalMPR(Ypred=score, testSet=folding['testSet'][['rowIndex', 'colIndex']].values + 1)
            mpr_list.append(mpr)

    mpr_mean = np.mean(np.array(mpr_list))
    mpr_std = np.std(np.array(mpr_list))
    return mpr_mean,mpr_std


#######################################################################################################################
# 2019-01-04

# 二维数组按列展开，返回对应元素值及其行、列索引
def meltArray(matr):
    # matr  np.array  二维形式
    # 按列展开，返回每一个元素及其对应索引 rowIndex, colIndex, value 的 list 形式
    # 最后返回 DataFrame

    cp_matr = np.copy(matr)

    m,n = cp_matr.shape
    rIndex = range(m)
    oneIndex = np.ones((m,),dtype=int)
    rowIndex = []
    colIndex = []
    value = []
    for i in range(n):
        rowIndex.extend(rIndex)
        cIndex = oneIndex * i
        colIndex.extend(cIndex.tolist())
        value.extend(cp_matr[:,i])

    df = pd.DataFrame(data={'rowIndex':rowIndex,'colIndex':colIndex,'value':value},columns=['rowIndex','colIndex','value'])
    return df

