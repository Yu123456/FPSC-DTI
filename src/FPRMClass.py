# coding:utf-8

# 特征投影关系预测模型
# 理论参见手稿 P20 - P20-5

import numpy as np
import scipy as sp
from scipy import linalg
import pandas as pd


# FPRM class
class FPRM:
    '''
    类属性：
    特征投影矩阵 R
    正则化参数 re_lambda

    类成员：（尽量与 sklearn 一致）
    训练函数 fit
    预测函数 predict_pair
    预测函数 predict_matrix
    '''

    # 初始化函数
    def __init__(self,
                 re_lambda = None):
        self.R = None
        self.re_lambda = re_lambda

    # 计算矩阵逆，如果矩阵不可逆，求其最小二乘逆
    def inverse_matrix(self,X):
        X_inv = None
        try:
            X_inv = linalg.inv(X)
        except:
            print('sigular matrix, compute linalg.lstsq solve inv(X)')
            n,m = X.shape
            Ematrix = np.eye(n,n,dtype=float)
            X_inv = linalg.lstsq(X,Ematrix)[0]
        finally:
            return X_inv

    # 训练函数
    def fit(self,X,Y,Delta):
        '''

        :param X:  特征矩阵，一行一个样本, N * m, N 个样本，m 个特征
        :param Y:  特征矩阵，一行一个样本， N * n, N 个样本，n 个特征
        :param Delta: 对应着特征矩阵 X,Y 的关系作用矩阵，维度 N * N，x_i R y_j^T = delta_{ij}
        :return: None
        '''

        x = np.copy(X)
        y = np.copy(Y)
        delta = np.copy(Delta)

        xTx = x.T.dot(x)   # x^T * x
        yTy = y.T.dot(y)   # y^T * y
        xTDy = x.T.dot(delta).dot(y)
        if self.re_lambda is None:
            # 没有正则项
            xTx_inv = self.inverse_matrix(xTx)
            yTy_inv = self.inverse_matrix(yTy)
            self.R = xTx_inv.dot(xTDy).dot(yTy_inv)
        else:
            # 没有正则项，需要求解 Sylvester 方程
            yTy_inv = self.inverse_matrix(yTy)
            B = self.re_lambda * yTy_inv
            Q = xTDy.dot(yTy_inv)
            # 注意，scipy.linalg.solve_sylvester 在 scipy 1.1.0 版本之后
            self.R = linalg.solve_sylvester(a=xTx,b=B,q=Q)

        return self


    # 预测函数，矩阵形式
    def predict_matrix(self,X,Y):
        '''

        :param X: 特征矩阵，一行一个样本
        :param Y: 特征矩阵，一行一个样本， X,Y 样本个数必须一致，特征维度可以不一致
        :return: 预测关系矩阵
        '''
        x = np.copy(X)
        y = np.copy(Y)

        return x.dot(self.R).dot(y.T)

    # 预测函数，pair 形式
    def predict_pair(self,X,Y):
        '''

        :param X: 特征矩阵，一行一个样本
        :param Y: 特征矩阵，一行一个样本， X,Y 样本个数必须一致，特征维度可以不一致
        :return: 返回对应样本数的向量
        '''
        n,m = X.shape
        val = np.zeros((n,1),dtype=float)
        for i in range(n):
            val[i] = X[i].dot(self.R).dot(Y[i].T)

        return val


# 特征投影关系隶属度类，继承 FPRM 类
# 会新添一些属性：
# ulabel  唯一标签 list
# dict_label  唯一标签，dict
# mu,sigma  隶属度函数参数，每一个标签对应一组参数
class FPRMmembership(FPRM):
    # 初始化函数
    def __init__(self,
                 re_lambda=None):
        FPRM.__init__(self, re_lambda=re_lambda)
        # 下面是新增属性
        self.ulabel = None     # 唯一标签，list
        self.dict_label = None    # 唯一标签，dict
        self.mu = None         # 对应类别标签的隶属度函数均值参数, dict 保存，易于参数值与标签对应
        self.sigma2 = None     # 对应类别标签的隶属度函数方差参数，dict 保存，易于参数值域标签对应

    # 训练函数
    def fit(self,X,Y,Delta):
        '''

        :param X:  特征矩阵，一行一个样本, N * m, N 个样本，m 个特征
        :param Y:  特征矩阵，一行一个样本， N * n, N 个样本，n 个特征
        :param Delta: 对应着特征矩阵 X,Y 的关系作用矩阵，维度 N * N，x_i R y_j^T = delta_{ij}
        :return: None
        '''

        delta = np.array(Delta)
        # 矩阵中不重复元素
        delta_list = delta.flatten()
        ulabel = np.unique(delta_list).tolist()   # 转成 list 形式

        # 计算 R
        FPRM.fit(self,X=X,Y=Y,Delta=Delta)
        # 计算训练集的预测值
        pred_delta = FPRM.predict_matrix(self,X=X,Y=Y)

        # 构造标签的索引字典，方便读取索引值
        D = {}
        mu = {}
        sigma2 = {}
        for i,val in enumerate(ulabel):
            D[val] = i      # 标签值为 val 对应 ulabel 的索引为 i, 即 ulabel[i] = val
            # 计算均值与方差
            mu[val] = np.mean(pred_delta[ Delta == val])
            sigma2[val] = np.var(pred_delta[Delta == val])
            # print('label : %s , mean : %.4f , var : %.4f' %(val,mu[val],sigma2[val]))

        self.ulabel = ulabel
        self.dict_label = D
        self.mu = mu
        self.sigma2 = sigma2
        return self

    # 预测函数，矩阵形式，返回指定标签的隶属度值
    def predict_score_matrix(self,X,Y,label):
        '''

        :param X: 特征矩阵，一行一个样本
        :param Y: 特征矩阵，一行一个样本， X,Y 样本个数必须一致，特征维度可以不一致
        :param label: 需要计算隶属度的标签
        :return:
        '''

        predict_value = X.dot(self.R).dot(Y.T)
        if label in self.ulabel:
            val2 = np.square(predict_value - self.mu[label])
            membership = np.exp(- val2 / (2.0 * self.sigma2[label]))
            return membership
        else:
            print('interaciton label error!')
            exit()


    #预测函数，pair 形式，返回指定标签的隶属度值
    def predict_score_pair(self,X,Y,label):
        '''

        :param X: 特征矩阵，一行一个样本
        :param Y: 特征矩阵，一行一个样本， X,Y 样本个数必须一致，特征维度可以不一致
        :param label: 指定需要返回标签的隶属度
        :return:
        '''

        n = X.shape[0]
        val = np.zeros((n,1),dtype=float)
        for i in range(n):
            val[i] = X[i].dot(self.R).dot(Y[i].T)

        if label in self.ulabel:
            val2 = np.square(val - self.mu[label])
            membership = np.exp(- val2 / (2.0 * self.sigma2[label]))
            return membership
        else:
            print('interaction label error!')
            exit()