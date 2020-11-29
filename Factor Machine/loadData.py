import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle

def movieGenre(filepath):
    movie=list(pd.read_csv(filepath)['movieId'])
    genre=list(pd.read_csv(filepath)['genres'])
    allGenre=[]
    movGen={}
    for id,gen in zip(movie,genre):
        movGen[int(id)]=gen.split('|')
        allGenre+=gen.split('|')
    return movGen,list(set(allGenre))

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


# 选取userID，movieID，genre one-hot作为特征
def featureMatrix(ratingPath,moviePath,allGenre,movGen):
    data1=pd.read_csv(ratingPath)
    data2=pd.read_csv(moviePath)
    data=pd.merge(data1,data2)
    data=np.array(data.drop(columns=['rating','timestamp','title','genres','userId']))
    gen=np.zeros([len(data),len(allGenre)])
    for sample in range(len(data)):
        for g in movGen[data[sample][0]]:
            gen[sample][allGenre.index(g)]=1
    # data=np.append(data,gen,axis=1)
    data=gen
    label=np.ones([len(data),1])
    data=np.append(data,label,axis=1)
    return data


def genreOneHot(genreList,allGenre):
    res=np.zeros(len(allGenre))
    for g in genreList:
        res[allGenre.index(g)]=1
    return list(res)


# 给每个用户选取负样本
def nSample(item_pool,userItemDict,allGenre,movGen,ratio):
    negativeSample=[]
    for userID in userItemDict:
        positiveMov=userItemDict[userID]
        positiveNum=len(positiveMov)
        negativeNum=0
        negativeMov=[]
        while negativeNum<ratio*positiveNum:
            item = random.choice(item_pool)
            if item not in positiveMov and item not in negativeMov:
                negativeMov=genreOneHot(movGen[item],allGenre)+[-1]
                negativeSample.append(negativeMov)
                negativeNum+=1
    return np.array(negativeSample)


def itemPool(filepath):
    file=pd.read_csv(filepath)
    pool=list(file['movieId'])
    return pool

def prepareData():
    ratingPath='/Users/user/git_repo/XinyiF-Movie-Recommender-System/movieLen/ratings.csv'
    moviePath='/Users/user/git_repo/XinyiF-Movie-Recommender-System/movieLen/movies.csv'
    mov,allGenre=movieGenre(moviePath)
    posiSample=featureMatrix(ratingPath,moviePath,allGenre,mov)
    userItemDict=userItem(ratingPath)
    item_pool=itemPool(ratingPath)
    negaSample=nSample(item_pool,userItemDict,allGenre,mov,0.5)
    data=np.append(posiSample,negaSample,axis=0)
    np.random.shuffle(data)
    X=data[:,:len(data[0])-1]
    y=data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    with open('data.pickle','wb') as f:
        pickle.dump(X_train,f)
        pickle.dump(X_test, f)
        pickle.dump(y_train, f)
        pickle.dump(y_test, f)
    return X_train, X_test, y_train, y_test

prepareData()




