import numpy as np
import pandas as pd
import cane
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from .atl_feat_eng import target_type, divide_dfs
from .atl_metrics import metrics_regression, metrics_classification, metrics_binary_classification

def pred_eval(train:pd.DataFrame, test:pd.DataFrame, target:str):
    
    X_train,X_test,y_train,y_test=divide_dfs(train,test,target)  
    
    list_estimators,rf,et=[100],[],[]
    pred_type, eval_metric=target_type(train, target)
    
    for estimators in list_estimators:
        
        if pred_type=='Reg': 
            regressor_RF = RandomForestRegressor(n_estimators=estimators, random_state=42)
            regressor_RF.fit(X_train, y_train)
            y_pred_rfr = regressor_RF.predict(X_test)
            RF_perf=pd.DataFrame(metrics_regression(y_test,y_pred_rfr),index=[0])
            RF_perf[['Estimators']]=estimators
            rf.append(RF_perf)
                    
            Reg_ET = ExtraTreesRegressor(n_estimators=estimators, random_state=42)
            Reg_ET.fit(X_train, y_train)
            y_pred_etr = Reg_ET.predict(X_test)
            ET_perf=pd.DataFrame(metrics_regression(y_test,y_pred_etr),index=[0])
            ET_perf[['Estimators']]=estimators
            et.append(ET_perf)
        
        elif pred_type=='Class': 
            classifier_RF = RandomForestClassifier(n_estimators=estimators, random_state=42)
            classifier_RF.fit(X_train, y_train)
            y_pred_rfc = classifier_RF.predict(X_test)
            RF_perf=pd.DataFrame(metrics_classification(y_test,y_pred_rfc),index=[0])
            RF_perf[['Estimators']]=estimators
            rf.append(RF_perf)
            
            classifier_ET = ExtraTreesClassifier(n_estimators=estimators, random_state=42)
            classifier_ET.fit(X_train, y_train)
            y_pred_etc = classifier_ET.predict(X_test)
            ET_perf=pd.DataFrame(metrics_classification(y_test,y_pred_etc),index=[0])
            ET_perf[['Estimators']]=estimators
            et.append(ET_perf)
        
    a,b=pd.concat(rf),pd.concat(et) 
         
    if pred_type=='Reg':
        x=pd.concat([a,b]) 
        x=x.sort_values(eval_metric, ascending=True)
    elif pred_type=='Class':
        x=pd.concat([a,b]) 
        x=x.sort_values(eval_metric, ascending=False)
    del x['Estimators']
    
    y,z=x.iloc[:1,:],x.iloc[1:2,:]
    metrics_final=(y+z)/2
    
    return metrics_final
