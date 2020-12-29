# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:56:10 2020

@author: Keshav Bansal
"""

import pandas as pd
import numpy as np
import xlsxwriter
#from seaborn import heatmap

train = pd.read_excel('Train_dataset.xlsx')
test = pd.read_excel('Test_dataset.xlsx')

wb=xlsxwriter.Workbook('pred res.xlsx')
sheet=wb.add_worksheet("Sheet 1")
for i in range(0,test.shape[0]):
    sheet.write(i,0,test.iloc[i,0])
    
x = train.iloc[:,:-1]
target = train.iloc[:,-1]

x.drop(x.columns[[0,1,3,4]], axis=1, inplace = True)
test.drop(test.columns[[0,1,3,4]], axis=1, inplace = True)
#heatmap(x.isnull(),yticklabels=False,cbar=False)

def missing(x):
    
    c=x.shape[1]
    col_dict={}
    col=[i for i in range(0,c)]  
    for i in range(0,c):
        count=x.isnull().iloc[:,i].sum()
        if count>0:
            col_dict[i]=count
    col_dict=dict(sorted(col_dict.items(), key=lambda item: item[1]))
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LinearRegression
    
    while len(col_dict)!=0:
        i=list(col_dict.keys())[0]
        df=x.iloc[:,list(set(col)-set(col_dict.keys()))]
        target=x.iloc[:,i]
        idx=[j for j in range(target.shape[0]-1,-1,-1) if target.isnull().iloc[j]]
        #x.drop(x.index[idx], inplace=True)
        #target.drop(target.index[idx], inplace=True)
        labelencoder=LabelEncoder()
        cat_cols=list(set(df.columns)-set(df._get_numeric_data().columns))
        for j in cat_cols:
            df[j]=labelencoder.fit_transform(df[j])
    
#        onehotencoder=OneHotEncoder(categories=[df.columns.get_loc(j) for j in cat_cols])
        onehotencoder=OneHotEncoder(categorical_features=[df.columns.get_loc(j) for j in cat_cols])
        df=onehotencoder.fit_transform(df).toarray()
        x_pred=[df[j] for j in idx]
        for j in idx:
            df=np.delete(df,j,axis=0)
        target.drop(target.index[idx], inplace=True)    
        if type(target.iloc[0])==str:
            dtc=DecisionTreeClassifier()
            dtc.fit(df,target)
            y_pred=dtc.predict(x_pred)
        else:
            regressor=LinearRegression()
            regressor.fit(df,target)
            y_pred=regressor.predict(x_pred)
        
        for j in range(len(idx)):
                x.iat[idx[j],i]=y_pred[j]
        del col_dict[i]


missing(x)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat_cols=list(set(x.columns)-set(x._get_numeric_data().columns))
for i in cat_cols:
    x[i]=le.fit_transform(x[i])
    test[i]=le.fit_transform(test[i])
    
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x=imputer.fit_transform(x)

#    
#nrows=x.shape[0]
#ncol=x.shape[1]
##indexes=[]
#count=0
#for i in range(0,ncol):
#    mean=x.mean(axis=0)[i]
#    std=x.std(axis=0)[i]
#    for j in range(nrows-1,-1,-1):
#        if (abs(x.iloc[j][i]-mean))/std > 1.5:
#            count+=1
#            #indexes.append(j)
#            x.drop(x.index[j],inplace=True)
#            nrows-=1
            
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[x.columns.get_loc(j) for j in cat_cols])
x=ohe.fit_transform(x).toarray()
ohe=OneHotEncoder(categorical_features=[test.columns.get_loc(j) for j in cat_cols])
test=ohe.fit_transform(test).toarray()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
test=sc.fit_transform(test)

#from sklearn.model_selection import KFold
#kf=KFold(n_splits=10,random_state=None)
#for train_index, test_index in kf.split(x):
#    print("Train:", train_index, "Validation:",test_index)
#    x_train, x_test = x[train_index], x[test_index] 
#    y_train, y_test = y[train_index], y[test_index]

#from sklearn.linear_model import LinearRegression
#regressor=LinearRegression()

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x,target)
y_pred=regressor.predict(test)

row=0
for i in y_pred:
    sheet.write(row,1,i)
    row+=1
wb.close()
#for i in train.columns:
#    print(train[i].isnull().value_counts())

#for i in test.columns:
    #print(test[i].nunique())
#z=pd.DataFrame([[1,2],[2,3],[4,5]])



