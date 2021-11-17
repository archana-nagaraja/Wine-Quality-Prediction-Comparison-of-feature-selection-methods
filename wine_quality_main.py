# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 08:00:57 2020

@author: Archana
Final Project : Wine Quality based on physiochemical properties
Goal: Compare Feature Selection methods

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
import apply_rfe as rfe
from sklearn.ensemble import  RandomForestRegressor
import copy  
from itertools import compress
from sklearn.preprocessing import StandardScaler

def apply_pearson_correlation(inp_data):
    try:
        print()
        print("Applying the Pearson Correlation method for feature-selection")
        # create the correlation matrix
        print("Creating correlation matrix")
        cor=inp_data.corr()
        mask = np.zeros_like(cor, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize = (12,12))
        hmap=sns.heatmap(cor,linewidths=.5,annot=True,fmt='.2g',\
                vmin=-1,mask=mask,cbar_kws= {'orientation': 'horizontal'})
        output_file = os.path.join(input_dir,'correlation_matrix.pdf')
        #plt.savefig(output_file) 
        plt.show()
        #Correlation with output variable
        cor_target = abs(cor['quality'])
        #select highly correlated features
        relevant_features=cor_target[(cor_target>0.15)&(cor_target<1)]
        print("Printing features identified using Pearson Correlation")
        print(relevant_features)
        flist=list(relevant_features.index)
        return flist
    except Exception as e:
        print(e)
        print('failed to apply Pearson correlation method')
        
def apply_lasso(inp_frame,flist):
    try:
        print()
        print("Applying the Lasso(L1 penalty) method for feature-selection")
        x=inp_frame[flist].values
        y=inp_frame[['quality']].values
        # LassoCV - to apply L1 penalty
        estimator=LassoCV(cv=5, normalize=True)
        sel=SelectFromModel(estimator,threshold=None)
        sel.fit(x,y.ravel())
        # get list of features selected
        ids=sel.get_support()
        lasso_list=list(compress(flist,ids))
        print("Number of features selected: "+str(len(lasso_list)))
        print("Printing list of features selected using Lasso"\
              "(L1 regularization) method - ")
        print(lasso_list)
        return lasso_list
    except Exception as e:
        print(e)
        print('failed to apply Lasso(L1 penalty) method for feature-selection')

def apply_random_forest(inp_frame,flist,method,fcount,fexcl):
    try:
        print()
        print("Applying random-forest regressor")
        # split the data-set into train & test
        inp_train,inp_test=train_test_split(inp_frame,train_size=0.5,\
                                            random_state=0)
        inp_train=inp_train.reset_index(drop=True)
        inp_test=inp_test.reset_index(drop=True)
        x_train=inp_train[flist].values
        y_train=inp_train[['quality']].values
        x_test=inp_test[flist].values
        y_test=inp_test[['quality']].values
        rfr=RandomForestRegressor(n_estimators=100,random_state=0)
        rfr.fit(x_train,y_train.ravel())
        inp_test=inp_test.assign(pred_quality=pd.Series(dtype=int))
        inp_test['pred_quality']=rfr.predict(x_test)
        inp_test['pred_quality']=inp_test['pred_quality'].round()
        pred_y=inp_test['pred_quality'].values
        # compute r2 and rmse to measure performance
        rmse=np.sqrt(mean_squared_error(inp_test['quality'],\
                    inp_test['pred_quality']))
        r2=r2_score(inp_test['quality'],inp_test['pred_quality'])
        print("r2 using RandomForest: "+str(round(r2,4)))
        print("rmse using RandomForest: "+str(round(rmse,4)))
        acc=accuracy_score(y_test,pred_y)
        print("Accuracy using RandomForest: "+str(round(acc,4)))
        n=results.shape[0]+1
        results.loc[n]=['RandomForest',method,round(acc,4),round(r2,4),\
                        round(rmse,4),fexcl,flist,fcount]
    except Exception as e:
        print(e)
        print('failed to apply random-forest regressor')

def apply_linear_regression(inp_frame,flist,method,fcount,fexcl):
    try:
        print()
        print("Applying linear regression model")
        # split the data-set into train & test
        inp_train,inp_test=train_test_split(inp_frame,train_size=0.5,\
                                            random_state=0)
        inp_train=inp_train.reset_index(drop=True)
        inp_test=inp_test.reset_index(drop=True)
        x_train=inp_train[flist].values
        y_train=inp_train[['quality']].values
        x_test=inp_test[flist].values
        y_test=inp_test[['quality']].values
        # scale the train-data
        x_train=StandardScaler().fit_transform(x_train)
        # initialize Lasso regressor
        lin_reg=LinearRegression(fit_intercept=True)
        # fit the train-data into regressor
        lin_reg.fit(x_train,y_train)
        # scale the test-data
        x_test=StandardScaler().fit_transform(x_test)
        inp_test=inp_test.assign(pred_quality=pd.Series(dtype=int))
        inp_test['pred_quality']=lin_reg.predict(x_test)
        inp_test['pred_quality']=inp_test['pred_quality'].round()
        # compute r2 and rmse to measure performance
        rmse=np.sqrt(mean_squared_error(inp_test['quality'],\
                    inp_test['pred_quality']))
        r2=r2_score(inp_test['quality'],inp_test['pred_quality'])
        print("r2 using LinearRegression: "+str(round(r2,4)))
        print("rmse using LinearRegression: "+str(round(rmse,4)))
        # calculating the accuracy-score 
        acc=lin_reg.score(x_test,y_test)
        print("Accuracy score using Linear Regression: "+\
              str(round(acc,4)))
        n=results.shape[0]+1
        results.loc[n]=['LinearRegression',method,round(acc,4),round(r2,4),\
                        round(rmse,4),fexcl,flist,fcount]
    except Exception as e:
        print(e)
        print('failed to apply linear regression')

def plot_shply(inp_data):
    try:
        print()
        print("Plotting results from Shapley method")
        input_dir = r'.'
        x=inp_data['feature_excluded'].values
        y=inp_data['accuracy'].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x,y,label='RandomForest')
        plt.plot(x,y)
        plt.title('Shapley Method - Feature-excluded vs accuracy')
        plt.xlabel('Feature-excluded')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45,fontsize='x-small')
        plt.legend()
        output_file = os.path.join(input_dir,'Plot2_ShapleyMethod_anagaraj.pdf')
        plt.savefig(output_file)
        plt.show()
    except Exception as e:
        print(e)
        print('failed while plotting results from Shackley method')

def plot_results(inp_data):
    try:
        print()
        print("Plotting results to compare the feature-selection methods.")
        input_dir = r'.'
        # split by method
        pearson=inp_data[inp_data['method']=='PearsonCoefficient'].\
                    copy(deep=True)
        pearson.reset_index(drop=True,inplace=True)
        pcount=pearson['features_used'][0]
        rfe=inp_data[inp_data['method']=='RFE'].copy(deep=True)
        rfe.reset_index(drop=True,inplace=True)
        rcount=rfe['features_used'][0]
        lassoCV=inp_data[inp_data['method']=='LassoCV'].copy(deep=True)
        lassoCV.reset_index(drop=True,inplace=True)
        lcount=lassoCV['features_used'][0]
        allfeatures=inp_data[inp_data['method']=='AllFeatures'].copy(deep=True)
        allfeatures.reset_index(drop=True,inplace=True)
        acount=allfeatures['features_used'][0]
        # plot line for Pearson Coefficient
        x=pearson['method'].values
        y=pearson['accuracy'].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x,y,label=str(pcount))
        plt.plot(x,y)
        # plot line for RFE
        x=rfe['method'].values
        y=rfe['accuracy'].values
        plt.scatter(x,y,label=str(rcount))
        plt.plot(x,y)
        # plot line for LassoCV
        x=lassoCV['method'].values
        y=lassoCV['accuracy'].values
        plt.scatter(x,y,label=str(lcount))
        plt.plot(x,y)
        # plot line for all-features
        x=allfeatures['method'].values
        y=allfeatures['accuracy'].values
        plt.scatter(x,y,label=str(acount))
        plt.title('Comparision of feature-selection methods - Method vs '\
                  'accuracy')
        plt.xlabel('Method')
        ax=plt.gca()
        ax.xaxis.set_label_coords(0,0)
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45,fontsize='x-small') 
        plt.legend(title='No. of features',frameon=True)
        output_file = os.path.join(input_dir,'Plot1_All_Results_anagaraj.pdf')
        plt.savefig(output_file)
        plt.show()
    except Exception as e:
        print(e)
        print('failed while plotting results to compare the feature-selection'\
              ' methods')   

try:
    print()
    print("Loading the red-wine dataset")
    input_dir = r'.'
    input_file = os.path.join(input_dir, 'winequality-red.csv')
    input_data_red=pd.read_csv(input_file,sep=";",header=0,names= \
                    ["fixed_acidity","volatile_acidity","citric_acid",\
                     "residual_sugar","chlorides","free_sulfur_dioxide",\
                     "total_sulfur_dioxide","density","ph","sulphates",\
                     "alcohol","quality"])
    # drop the index column
    input_data_red.reset_index(level=0,inplace=True,drop=True)
    #print(input_data_red['quality'].value_counts())
    #input_data_red.to_csv('./mycsv_input_data_red.csv')
    
    # global dataframe to compile results
    global results
    results=pd.DataFrame(dtype=object)
    results=results.assign(model=pd.Series(dtype=str))
    results=results.assign(method=pd.Series(dtype=str))
    results=results.assign(accuracy=pd.Series(dtype=float))
    results=results.assign(r2=pd.Series(dtype=float))
    results=results.assign(rmse=pd.Series(dtype=float))
    results=results.assign(feature_excluded=pd.Series(dtype=str))
    results=results.assign(feature_list=pd.Series(dtype=object))
    results=results.assign(features_used=pd.Series(dtype=int))
    
    # list of all 11 features
    all_list=["fixed_acidity","volatile_acidity","citric_acid",\
                     "residual_sugar","chlorides","free_sulfur_dioxide",\
                     "total_sulfur_dioxide","density","ph","sulphates",\
                     "alcohol"]
    
    
    # Feature-Selection method-1: apply Pearson Correlation method
    print()
    print("Feature Selection method-1: Pearson Correlation")
    pearson_list=apply_pearson_correlation(input_data_red)
    
    apply_random_forest(input_data_red,pearson_list,'PearsonCoefficient',\
                        len(pearson_list),'N/A')
    """
    apply_linear_regression(input_data_red,pearson_list,'PearsonCoefficient',\
                        len(pearson_list),'N/A')
    """   
    # Feature-Selection method-2: Recursive-Feature-Elimination
    print()
    print("Feature Selection method-2: Recursive Feature Elimination ")
    rfe_list,rfe_count=rfe.apply_rfe(input_data_red)
    apply_random_forest(input_data_red,rfe_list,'RFE',rfe_count,'None')
    #apply_linear_regression(input_data_red,rfe_list,'RFE',rfe_count,'N/A')
    
    # Feature-Selection method-3: LassoCV
    print()
    print("Feature Selection method-3: L1 regularization - LassoCV")
    lasso_list=apply_lasso(input_data_red,all_list)
    
    apply_random_forest(input_data_red,lasso_list,'LassoCV',\
                        len(lasso_list),'N/A')
    """
    apply_linear_regression(input_data_red,lasso_list,'LassoCV',\
                        len(lasso_list),'N/A')
    """
    
    # drop one feature at a time and apply models to determine 
    # r2/rmse/accuracy metrics
    print()
    print("Feature Selection method-4: Shapley method")
    for i in range(0,len(all_list)):
        temp_list=copy.deepcopy(all_list)
        temp_list.remove(all_list[i])
        """
        # invoke linear-regression
        print("Dropped feature: "+all_list[i]+" - Applying Linear regression")
        apply_linear_regression(input_data_red,temp_list,'Shapley-method',\
                        10,all_list[i])
        """
        # invoke Random-Forest 
        print("Dropped feature: "+all_list[i]+" - Invoking RandomForest for "\
              "regression")
        apply_random_forest(input_data_red,temp_list,'Shapley-method',10,\
                            all_list[i])
        
    # Use all 11 features to compare the accuracy
    print("Using all 11 features to compare accuracy of prediction")
    
    apply_random_forest(input_data_red,all_list,'AllFeatures',\
                        11,'None')
    """
    apply_linear_regression(input_data_red,all_list,'AllFeatures',\
                        11,'None')
    """
    results.to_csv('./random_forest_results.csv')
      
    # VISUALIZATION
    # split the results as needed for plotting
    #plot#1
    c1=results['method']!='Shapley-method'
    others=results[c1].copy(deep=True)
    others.reset_index(drop=True,inplace=True)
    # invoke method to plot
    plot_results(others)
    #plot#2
    c2=((results['method']=='Shapley-method') | \
        (results['method']=='AllFeatures'))
    shply=results[c2].copy(deep=True)
    shply.reset_index(drop=True,inplace=True)
    # invoke method to plot
    plot_shply(shply)
except Exception as e:
        print(e)
        print('failed in the main procedure')

