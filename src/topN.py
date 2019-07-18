# coding:utf-8

import numpy as np
import pandas as pd
from copy import deepcopy


# 计算 top N
def topN(df_score,list_dID,list_tID,list_tops,df_checked):
    '''

    :param df_score: 预测得分 DataFrame, 含（score,target 序号，drug 序号）
    :param list_dID:  drug ID
    :param list_tID:  target ID
    :param list_tops: top N 取值
    :param df_checked: 已验证 interaction pair, DataFrame 格式,(Drug ID, Target ID, Checked)
    :return:
    '''

    # 深度拷贝，防止对原始数据的更改
    df_s = deepcopy(df_score)
    dID = np.array(deepcopy(list_dID))
    tID = np.array(deepcopy(list_tID))
    tops = np.array(deepcopy(list_tops))
    df_c = deepcopy(df_checked)

    # 按照 rowIndex, colIndex 序号，将 target ID, drug ID 添加到 df_s 中，新增两列
    # rowIndex --> target, colIndex --> drug
    df_s['Drug ID'] = dID[df_s['colIndex'].values]
    df_s['Target ID'] = tID[df_s['rowIndex'].values]

    # 对 df_s 按照 score 从大到小排序
    # 特别注意，排序后仍然保持了原来的 index，所以如果按行取排序后的数据，用绝对索引 df_top.iloc
    df_top = df_s.sort_values(by=['score'],ascending=False)
    topChecked = pd.merge(df_top,df_c,on=['Drug ID','Target ID'],how='left')

    # 计算 top N
    ts = []
    start = 0
    ser = topChecked['Checked']
    for i,stop in enumerate(tops):
        value = ser[start:stop]
        ts.append(np.sum(value == 'T'))

    return ts