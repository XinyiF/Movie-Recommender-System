import numpy as np
import pickle
import pandas as pd
from loadData import *

# {item:[0,0,1,0,0,0,...]}
# 用户点击记为1
def userItemClic(userItemDict,itemUserDict):
    rating={}
    for item in itemUserDict:
        rating[item]=np.zeros(max(userItemDict)+1)
        for usr in itemUserDict[item]:
            rating[item][usr]=1
    return rating


# 向量相似度
# 欧式距离相似度
def ecludSim(A,B):
    return 1.0/(1.0 + np.linalg.norm(A - B))

# 余弦相似度
def cosSim(A,B):
    A_B = sum(A[i] * B[i] for i in range(len(A)))
    cos=A_B/(np.linalg.norm(A)*np.linalg.norm(B))
    return 0.5+0.5*cos

# 皮尔逊相似度（向量减去平均值后做余弦相似度）
def pearSim(A,B):
    meanA=np.mean(A)
    meanB = np.mean(B)
    A_B=sum((A[i]-meanA)*(B[i]-meanB) for i in range(len(A)))
    pear=A_B / (np.linalg.norm(A-meanA) * np.linalg.norm(B-meanB))
    return 0.5+0.5*pear

def sim(rating):
    simMatrix=np.zeros([max(rating)+1,max(rating)+1])
    for i in rating:
        for j in rating:
            if i == j:
                simMatrix[i][j]=float('inf')
            else:
                simMatrix[i][j]=ecludSim(rating[i],rating[j])
    return simMatrix

def main():
    userItemDict=userItem('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
    itemUserDict=itemUser('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
    rating=userItemClic(userItemDict,itemUserDict)
    s=sim(rating)
    with open('simMatrix.pickle','wb') as f:
        pickle.dump(s,f)

if __name__ == "__main__":
    main()

