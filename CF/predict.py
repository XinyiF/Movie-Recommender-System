import numpy as np
import pickle
from loadData import *

def kernal(v1,v2):
    return sum(v1[i]*v2[i] for i in range(len(v1)))


def predict(historyRating,simMatrix):
    res=np.zeros_like(historyRating)
    for item in range(len(historyRating)):
        score,count=0,0
        for item2 in range(len(historyRating)):
            if item!=item2 and historyRating[item2]!=0:
                score+=historyRating[item2]*simMatrix[item][item2]
                count+=simMatrix[item][item2]
        if count!=0:
            score/=count
        res[item]=score
    return res

def recoItemCF(predictRate,recoNum):
    idx=np.argsort(predictRate)[::-1]
    res=idx[:recoNum]
    return res

# 数据太稀疏
def main():
    with open('simMatrix.pickle','rb') as f:
        simMatrix=pickle.load(f)
    rating=userItemClic('/Users/user/git_repo/Movie-Recommender-System/data/input_txt/CoMoDa.csv')
    train,test=rating[:250],rating[250:]
    ctr=[]
    recoNum=5
    for user in test:
        predRate=predict(user,simMatrix)
        itemIdx=recoItemCF(predRate,recoNum)
        count=0
        for idx in itemIdx:
            if user[idx]!=0:
                count+=1
        ctr.append(count/recoNum)
    print('平均ctr:',sum(ctr)/len(ctr))

if __name__ == "__main__":
    main()
