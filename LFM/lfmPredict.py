import pickle
from loadData import *
import random

def predict(itemID,P,Q,userID,predNum):
    res=[]
    pred={}
    for item in itemID:
        pred[sum(P[userID]*Q[item])]=item
    score=sorted(list(pred.keys()))[::-1]
    for i in range(predNum):
        res.append(pred[score[i]])
    return res


with open('PQ.pickle','rb') as f:
    P=pickle.load(f)
    Q=pickle.load(f)

userItemDict=userItem('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
itemUserDict=itemUser('/Users/user/git_repo/Movie-Recommender-System/movieLen/ratings.csv')
userID=list(userItemDict.keys())
itemID=list(itemUserDict.keys())
test_user= random.sample(userID, 50)
ctr=[]
for user in test_user:
    actual=userItemDict[user]
    pred = predict(itemID, P, Q, user, 20)
    ctr.append(len(set(actual)&set(pred))/len(pred))
print('CTR=',sum(ctr)/len(ctr)*100,'%')





