import pandas as pd
import numpy as np
import random
import pickle

# {user:[items]} dict
def userItem(filepath):
    dic={}
    file=pd.read_csv(filepath)
    user_item=pd.DataFrame(file,columns=['userId','movieId'])
    data=list(user_item.groupby('userId'))
    for usr in data:
        userid=usr[0]
        dic[userid]=list(usr[1]['movieId'])
    return dic

def itemUser(filepath):
    dic={}
    file=pd.read_csv(filepath)
    user_item=pd.DataFrame(file,columns=['userId','movieId'])
    data=list(user_item.groupby('movieId'))
    for item in data:
        movieId=item[0]
        dic[movieId]=list(item[1]['userId'])
    return dic


# 流行度定义为看过该电影的人数和总人数的比例
def popularity(userItemDict,itemUserDict):
    dic,userNum={},len(userItemDict)
    for item in itemUserDict:
        dic[item]=len(itemUserDict[item])/userNum
    return dic

def itemPool(filepath):
    file=pd.read_csv(filepath)
    pool=list(file['movieId'])
    return pool

# 对每个用户进行正负样本采样
def nSample(item_pool,userItemDict,ratio):
    print('开始采集样本...')
    dic={}
    # 记录正样本
    for user in userItemDict:
        dic[user]={}
        for item in userItemDict[user]:
            dic[user][item]=1
    # 采集和正样本平衡的负样本
    # 负样本被采集到的几率和流行率成正比
    for user in dic:
        # 正样本个数
        numPositive=len(dic[user])
        numNegative=0
        while numNegative<numPositive*ratio:
            item=random.choice(item_pool)
            if item not in dic[user]:
                dic[user][item]=0
                numNegative+=1
    return dic

# 初始化每个用户的P，每个item的Q
def initPQ(userItemDict,itemUserDict,K):
    print('开始初始化...')
    P,Q={},{}
    for user in userItemDict:
        P[user]=np.random.random(K)
    for item in itemUserDict:
        Q[item]=np.random.random(K)
    return P,Q

def lfm(P,Q,sample,K,maxIter=20,alpha=0.02,lamb=0.01):
    for it in range(maxIter):
        for user in sample:
            for item in sample[user]:
                eui = sample[user][item] - sum(P[user][i] * Q[item][i] for i in range(K))
                for f in range(K):
                    P[user][f] += alpha * (Q[item][f] * eui - lamb* P[user][f])
                    Q[item][f] += alpha * (P[user][f] * eui - lamb * Q[item][f])
        alpha*=0.9
        print('第',it,'次迭代')
    return P,Q





def main():
    userItemDict=userItem('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
    itemUserDict=itemUser('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
    pop=popularity(userItemDict,itemUserDict)
    sortedPop=dict(sorted(pop.items(),key=lambda item:item[1]))
    item_pool=itemPool('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
    sample=nSample(item_pool,userItemDict,2)
    P,Q=initPQ(userItemDict,itemUserDict,3)
    P,Q=lfm(P,Q,sample,3)
    with open('PQ.pickle','wb') as f:
        pickle.dump(P,f)
        pickle.dump(Q,f)









