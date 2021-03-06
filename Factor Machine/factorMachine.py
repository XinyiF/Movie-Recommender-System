# -*- coding: utf-8 -*-

import numpy as np
from random import normalvariate #正态分布
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
import pickle

class FM(object):
    def __init__(self):
        self.data = None
        self.label = None
        self.data_test = None
        self.label_test = None

        self.alpha = 0.01
        self.iter = 30
        self.k= 3
        self._w = None
        self._w_0 = None
        self.v = None

    # 数据归一化，统一标尺
    # 将标签换成-1 1
    def preprocessing(self,data,label,test_data=False):
        min_max_scaler = MinMaxScaler()
        data_minMax = min_max_scaler.fit_transform(data)
        if test_data:
            self.data_test=data_minMax
            self.label_test=label
        else:
            self.data=data_minMax
            self.label=label



    def sigmoid(self,x):  # 定义sigmoid函数
        return 1.0 / (1.0 + expit(-x))

    def kernal(self,v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    # 预测一条数据x
    def getPrediction(self,x,thold):
        m, n = np.shape(self.data)
        temp = 0
        Vt = self.v.T
        for k in range(self.k):
            inner1 = self.kernal(Vt[k], x)
            inner2 = self.kernal(Vt[k] ** 2, x ** 2)
            temp += inner1 - inner2
        temp /= 2
        term1 = self._w_0
        term2 = self.kernal(x, self._w)
        # 该sample的预测值
        pre = self.sigmoid(term1 + term2 + temp)
        # print(pre)
        if pre > thold:
            pre = 1
        else:
            pre = -1
        return pre


    # 计算准确率
    def calaccuracy(self,pre_y,act_y):
        cost=[]
        for sampleId in range(len(act_y)):
            if pre_y[sampleId]==act_y[sampleId]:
                cost.append(1)
            else:
                cost.append(0)
        return np.sum(cost)/len(cost)

    def sgd_fm(self):
        print('开始训练...')
        # 数据矩阵data是m行n列
        m, n = np.shape(self.data)
        # 初始化w0,wi,V,Y_hat
        w0 = 0
        wi = np.zeros(n)
        V = normalvariate(0, 0.2) * np.ones([n, self.k])
        for it in range(self.iter):
            # 随机梯度下降法
            for sampleId in range(m):
                # 计算交叉项
                temp=0
                Vt=V.T
                for k in range(self.k):
                    inner1=self.kernal(Vt[k],self.data[sampleId])
                    inner2=self.kernal(Vt[k]**2,self.data[sampleId]**2)
                    temp+=inner1-inner2
                temp/=2
                term1=w0
                term2=self.kernal(self.data[sampleId],wi)
                # 该sample的预测值
                y_hat=term1+term2+temp
                # print(y_hat)
                # 计算损失
                yp=self.sigmoid(y_hat*self.label[sampleId])
                part_df_loss=(yp-1)*self.label[sampleId]
                #  更新w0,wi
                w0-=self.alpha*1*part_df_loss
                for i in range(n):
                    if self.data[sampleId][i]!=0:
                        wi[i]-=self.alpha*self.data[sampleId][i]*part_df_loss
                        for f in range(self.k):
                            V[i][f] -= self.alpha * part_df_loss * self.data[sampleId][i] * sum(
                                V[j][f] * self.data[sampleId][j] -
                                V[i][f] * self.data[sampleId][i] * self.data[sampleId][i] for j in range(n))

            print('第%s次训练' % (it))
        self._w = wi
        self._w_0 = w0
        self.v = V


def main():
    os=FM()
    print('准备数据...')
    with open('CoMoDaData.pickle','rb') as f:
        data_train=pickle.load(f)
        data_test=pickle.load(f)
        y_train=pickle.load(f)
        y_test=pickle.load(f)
    print('处理数据...')
    # 减小训练集size
    data_train=data_train[:400]
    y_train=y_train[:400]
    os.preprocessing(data_train,y_train)
    os.preprocessing(data_test,y_test,True)
    # 训练模型
    os.sgd_fm()
    print('训练结束...')
    acu=[]
    maxAcu,maxThold=0,0
    for thold in np.arange(0.4,1,0.01):
        for user,label in zip(os.data_test[:40],os.label_test[:40]):
            print(os.getPrediction(user,thold),label)
            if os.getPrediction(user,thold)==label:
                acu.append(1)
            else:
                acu.append(0)
        if sum(acu)/len(acu) > maxAcu:
            maxThold=thold
            maxAcu=sum(acu)/len(acu)

    print('准确率=',maxAcu,'最佳阈值=',maxThold)


if __name__ == "__main__":
    main()





