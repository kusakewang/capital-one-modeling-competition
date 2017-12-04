# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:29:30 2016

@author: wenja
"""

import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

#Dealing with the raw data,according to the proposal we have:
def convert(data):
    categorical=np.array([9,11,12,13,14,15,16,18,20,21,\
    22,23,26,29,30,32,34,35,36,37,38,40,44,46,48,50,52,\
    55,57,58,59,60,62,64,65,66,68])-1
    for i in categorical:
        number = preprocessing.LabelEncoder()
        number.fit(data.iloc[:,i])
        data.iloc[:,i] = number.fit_transform(data.iloc[:,i])
    return data
    
test1cov=convert(rawdata)
#choose these varibales to train the random forest
x=np.array([ 5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,\
       23, 24, 25, 26, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,\
       42, 43, 44, 45, 46, 47, 48, 50, 52, 54, 55, 57, 58, 59, 60,\
       62, 64, 65, 66, 68, 69])-1
testX=np.array(test1cov.iloc[:,x])   
testY=np.array(test1cov.iloc[:,2])   
#Extract the traning model we stored
import pickle
with open('clfnew1.pkl', 'rb') as f:
    clf= pickle.load(f)  
##compute the proba    
p_fraud=clf.predict_proba(testX)[:,1]
Decline=np.zeros(len(testY),dtype=str)
#Through the traning we choose the cut-off as 0.65
Decline[np.nonzero((p_fraud>=0.65))]='Y'
Decline[np.nonzero((p_fraud<0.65))]='N'  
Decision=np.dataframe(np.c_[test1cov.iloc[:,0],p_fraud,Decline])
Decision.to_csv('Decision.csv',sep='|',header=('Auth_ID','P_FRAUD','DECLINE'))



  



