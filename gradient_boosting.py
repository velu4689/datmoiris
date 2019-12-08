# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:23:11 2018

@author: velmurugan.m
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

irisg=pd.read_csv('..\data\iris.csv')
irisg['is_train'] = np.random.uniform(0, 1, len(irisg)) <= .80
traing, testg = irisg[irisg['is_train']==True], irisg[irisg['is_train']==False]
features = irisg.columns[:4]

clfg = GradientBoostingClassifier(n_estimators=2)

yg = pd.factorize(traing['Species'])[0]
clfg.fit(traing[features], yg)
zg = pd.factorize(testg['Species'])[0]
xg=clfg.predict(testg[features])
pd.crosstab(testg['Species'], xg, rownames=['Actual Species'], colnames=['Predicted Species'])
list(zip(traing[features], clfg.feature_importances_))
print(accuracy_score(zg,xg))
