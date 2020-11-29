import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle

def featureMatrix(path,userFeature,itemFeature):
    file=pd.read_csv(path)
    # 将缺失数据替换成0
    file=file.replace(-1,0)
    user_item=np.array(pd.DataFrame(file,columns=['userID','itemID']))
    data=np.array(pd.DataFrame(file,columns=userFeature+itemFeature))
    oneHot=[]
    for sample in user_item:
        oneHot.append(UserItemOneHot(sample[0],sample[1],path))
    data=np.append(np.array(oneHot),data,axis=1)
    data=np.append(data,np.ones([len(data),1]),axis=1)
    return data

def UserItemOneHot(userID,itemID,path):
    MaxUserID = max(list(pd.read_csv(path)['userID']))
    MaxItemID=max(list(pd.read_csv(path)['itemID']))
    res=np.zeros(MaxItemID+MaxUserID+2)
    res[userID]=1
    res[MaxUserID+1+itemID]=1
    return list(res)

def userItemDict(path):
    file = pd.read_csv(path)
    data=list(file.groupby('userID'))
    res={}
    for usr in data:
        res[usr[0]]=list(usr[1]['itemID'])
    return res

def negativeSample(path,userFeature,itemFeature,ratio):
    file = pd.read_csv(path)
    item_pool=np.array(file['itemID'])
    user_pool=np.array(file['userID'])
    userRela=np.array(pd.DataFrame(file,columns=userFeature))
    itemRela=np.array(pd.DataFrame(file,columns=itemFeature))
    # 随机选取userID和itemID，如没有行为则作为负样本
    userItem=userItemDict(path)
    negaNum=0
    negaSample=[]
    while negaNum<ratio*len(item_pool):
        usr,item=np.random.randint(len(user_pool)),np.random.randint(len(item_pool))
        if item_pool[item] not in userItem[user_pool[usr]]:
            oneHot=UserItemOneHot(usr,item,path)
            temp=oneHot+list(userRela[usr])+list(itemRela[item])+[-1]
            negaSample.append(temp)
            negaNum+=1
    return np.array(negaSample)










path='/Users/user/git_repo/Movie-Recommender-System/CoMoDa/CoMoDa.csv'

userFeature=['age','sex','city','country','location','social','endEmo','dominantEmo','mood','physical']
itemFeature=['director','movieCountry','movieLanguage','movieYear','genre1','genre2','genre3','actor1','actor2','actor3','budget']
# data=featureMatrix(path,userFeature,itemFeature)
# with open('positiveData.pickle','wb') as f:
#     pickle.dump(data,f)
# nega=negativeSample(path,userFeature,itemFeature,1)
# with open('negativeData.pickle','wb') as f:
#     pickle.dump(nega,f)