# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:45:04 2020

@author: Archana
"""


from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def apply_rfecv(inp_frame):
    try:
        print()
        print("Applying Recursive Feature EliminationCV with DecisionTrees")
        x=inp_frame[["fixed_acidity","volatile_acidity","citric_acid",\
                     "residual_sugar","chlorides","free_sulfur_dioxide",\
                     "total_sulfur_dioxide","density","ph","sulphates",\
                     "alcohol"]].values
        y=inp_frame[['quality']].values
        """
        # use rfe with logistic-regression to find features
        model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=7000)
        """
        # use decision-trees to find features
        model=DecisionTreeRegressor(random_state=0)
        
        rfe=RFECV(estimator=model,cv=5)
        fit=rfe.fit(x,y.ravel())
        tmp=fit.support_
        num=fit.n_features_
        names=np.array(["fixed_acidity","volatile_acidity","citric_acid",\
                     "residual_sugar","chlorides","free_sulfur_dioxide",\
                     "total_sulfur_dioxide","density","ph","sulphates",\
                     "alcohol"])
        list=[]
        for i in range(0,names.size):
            if tmp[i]==True:
                list.append(names[i])
        print("Number of features selected: "+str(num))
        print("Feature Ranking: %s" % (fit.ranking_))
        print("List of features selected using RFE- ")
        print(list)
        return list,num
    except Exception as e:
        print(e)
        print('failed to apply RFECV with DecisionTrees')
        
def apply_rfe(inp_frame):
    try:
        print()
        print("Applying Recursive Feature Elimination with DecisionTrees")
        x=inp_frame[["fixed_acidity","volatile_acidity","citric_acid",\
                     "residual_sugar","chlorides","free_sulfur_dioxide",\
                     "total_sulfur_dioxide","density","ph","sulphates",\
                     "alcohol"]].values
        y=inp_frame[['quality']].values
        # use rfe with logistic-regression to identify top 6 features
        """
        model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=7000)
        """
        # use decision-trees to find features
        model=DecisionTreeRegressor(random_state=0)
        
        rfe=RFE(estimator=model,n_features_to_select=7)
        fit=rfe.fit(x,y.ravel())
        tmp=fit.support_
        num=fit.n_features_
        names=np.array(["fixed_acidity","volatile_acidity","citric_acid",\
                     "residual_sugar","chlorides","free_sulfur_dioxide",\
                     "total_sulfur_dioxide","density","ph","sulphates",\
                     "alcohol"])
        list=[]
        for i in range(0,names.size):
            if tmp[i]==True:
                list.append(names[i])
        print("Number of features selected: "+str(num))
        print("Feature Ranking: %s" % (fit.ranking_))
        print("List of features selected using RFE- ")
        print(list)
        return list,num
    except Exception as e:
        print(e)
        print('failed to apply RFE with DecisionTrees')