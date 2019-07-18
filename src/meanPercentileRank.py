# -*- coding:utf-8 -*-

from collections import Counter
from collections import OrderedDict
import numpy as np

def percentileRank(x):
    '''
    percentile-ranking
    :param x:  score (vector or list type)
    :return:
    '''

    ar = np.array(x,dtype=float)
    c = Counter(ar)
    rx = OrderedDict(sorted(c.items(),key=lambda t:t[0], reverse=False))
    smaller = np.cumsum(np.r_[0,rx.values()])[range(len(rx.values()))]
    larger = list(reversed(np.cumsum(np.r_[0,list(reversed(rx.values()))])))[1:]
    rxpr = np.array(smaller,dtype=float)/(np.array(smaller,dtype=float) + np.array(larger,dtype=float))
    res = np.copy(ar)
    for i in range(len(ar)):
        res[i] = rxpr[rx.keys().index(ar[i])]

    pr = 1.0-res
    return(pr)

# 计算 Mean Percentile Rank
def evalMPR(Ypred,testSet):
    '''
    计算 target-drug score matrix 的 Mean Percentile Rank
    :param Ypred:  target * drug score matrix
    :param testSet:  两列的矩阵，第一列为 rowIndex 列，即 target 索引，
                      第二列为 colIndex 列，即 drug 索引
    :return MPR: testSet 的 Mean Percentile Rank 值
    '''

    tSet = np.array(testSet,dtype=int)
    testCol = np.sort(np.unique(tSet[:, 1]))
    MPR = 0.0
    for i in testCol:
        ypredi = Ypred[:, i - 1]  # 索引从 0 开始
        prr = percentileRank(ypredi)
        idxTest = tSet[tSet[:, 1] == i, 0]
        MPR = MPR + np.mean(prr[idxTest - 1])

    MPR = MPR/len(testCol)
    return(MPR)