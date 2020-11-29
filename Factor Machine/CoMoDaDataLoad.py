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
    # oneHot=[]
    # for sample in user_item:
    #     oneHot.append(UserItemOneHot(sample[0],sample[1],path))
    # data=np.append(np.array(oneHot),data,axis=1)
    # data=np.append(user_item,data,axis=1)
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
            # oneHot=UserItemOneHot(usr,item,path)
            # temp=oneHot+list(userRela[usr])+list(itemRela[item])+[-1]
            temp =list(userRela[usr]) + list(itemRela[item]) + [-1]
            negaSample.append(temp)
            negaNum+=1
    return np.array(negaSample)

# path='/Users/user/git_repo/Movie-Recommender-System/CoMoDa/CoMoDa.csv'
#
# userFeature=['age','sex','city','country','daytype','season','location','weather','social','endEmo','dominantEmo','mood','physical','decision','interaction']
# itemFeature=['director','movieCountry','movieLanguage','movieYear','genre1','genre2','genre3','actor1','actor2','actor3','budget']
# neg=negativeSample(path,userFeature,itemFeature,1)
# pos=featureMatrix(path,userFeature,itemFeature)
# with open('positiveData.pickle','wb') as f:
#     pickle.dump(pos,f)
# with open('negativeData.pickle','wb') as f:
#     pickle.dump(neg,f)


def prepareData():
    with open('positiveData.pickle','rb') as f:
        pos=pickle.load(f)
    with open('negativeData.pickle', 'rb') as f:
        neg = pickle.load(f)
    data=np.append(pos,neg,axis=0)
    np.random.shuffle(data)
    # data=data
    X=data[:,:len(data[0])-1]
    y=data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    with open('CoMoDaData.pickle','wb') as f:
        pickle.dump(X_train,f)
        pickle.dump(X_test, f)
        pickle.dump(y_train, f)
        pickle.dump(y_test, f)
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test=prepareData()

