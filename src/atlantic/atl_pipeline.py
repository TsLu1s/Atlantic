import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from .atl_feat_eng import *
from .atl_feat_selection import *
from .atl_metrics import *
from .atl_performance import * 
from .atl_processing import *

h2o.init()

"""
import cane
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
"""

#############################################################################################################################################
###########################################################    Atlantic Pipeline   ##########################################################
#############################################################################################################################################

def fit_processing(Dataset:pd.DataFrame,
                   target:str, 
                   Split_Racio:float,
                   total_vi:float=0.98,
                   h2o_fs_models:int =7,
                   encoding_fs:bool=True,
                   vif_ratio:float=10.0):
    
    pred_type, eval_metric=target_type(Dataset, target) ## Prediction Contextualization
    if pred_type=='Class': Dataset[target]=Dataset[target].astype(str)
    
    Dataframe_=Dataset.copy()
    Dataset_=Dataframe_.copy()

############################## Validation Procidment ##############################
    Dataset_=remove_columns_by_nulls(Dataset_, 99.99)
    sel_cols= list(Dataset_.columns)
    sel_cols.remove(target)
    sel_cols.append(target) 
    Dataset_=Dataset_[sel_cols] ## target -> Last Column Index

    train, test= split_dataset(Dataset_,Split_Racio)

    train_test_=train.copy(),test.copy()

    train=del_nulls_target(train,target) ## Delete target Null Values
    test=del_nulls_target(test,target) ## Delete target Null Values
    
############################## Feature Engineering Date Column ##############################

    train=engin_date(train)
    test=engin_date(test)

############################## Feature Selection ##############################
    
    sel_cols, Sel_Imp =feature_selection_h2o(train,target,total_vi,h2o_fs_models,encoding_fs)
    print('Selected Columns:', sel_cols)

    train=train[sel_cols]
    test=test[sel_cols]

############################## Encoding Method Selection ##############################   

    enc_method=Select_Encoding_Method(train,test,target,pred_type,eval_metric)

############################## NULL SUBSTITUTION + ENCODING APLICATION ##############################

    if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
        train, test,imp_method=null_substitution_method(train,test,target,enc_method,pred_type,eval_metric)
    else:
        ## Encoding Method
        imp_method='Undefined'
        if enc_method=='Encoding Version 1':
            train, test=encoding_v1(train, test,target)
        elif enc_method=='Encoding Version 2':
            train, test=encoding_v2(train, test,target)
        elif enc_method=='Encoding Version 3':
            train, test=encoding_v3(train, test,target)
        elif enc_method=='Encoding Version 4':
            train, test=encoding_v4(train, test,target)
    
        print('    ')   
        print('There are no missing values in the Input Data')    
        print('    ') 
        
############################## VARIANCE INFLATION FACTOR (VIF) APPLICATION ##############################
    
    train, test=vif_performance_selection(train,test,target)

############################## Fit Procediment ##############################

    Dataframe=Dataframe_.copy()
    Dataframe=del_nulls_target(Dataframe,target) ## Delete target Null Values 
    Dataframe=engin_date(Dataframe)
    Dataframe=Dataframe[list(train.columns)]
    train_df,test_df=split_dataset(Dataframe,Split_Racio)
    train_df,n_cols=Dataframe.copy(),num_cols(Dataframe,target)
    
    n_cols,c_cols=num_cols(train_df,target),cat_cols(train_df,target) 

    if enc_method=='Encoding Version 1':
        if len(n_cols)>0:
            scaler,n_cols= fit_StandardScaler(train_df,target)
            train_df[n_cols] = scaler.transform(train_df[n_cols])
            test_df[n_cols] = scaler.transform(test_df[n_cols])    
        if len(c_cols)>0:
            fit_pre=fit_IDF_Encoding(train_df,target)
            train_df=transform_IDF_Encoding(train_df,fit_pre)
            test_df=transform_IDF_Encoding(test_df,fit_pre)

    elif enc_method=='Encoding Version 2':
        if len(n_cols)>0:
            scaler,n_cols=fit_MinmaxScaler(train_df,target)
            train_df[n_cols] = scaler.transform(train_df[n_cols])
            test_df[n_cols] = scaler.transform(test_df[n_cols])   
        if len(c_cols)>0:
            fit_pre=fit_IDF_Encoding(train_df,target)
            train_df=transform_IDF_Encoding(train_df,fit_pre)
            test_df=transform_IDF_Encoding(test_df,fit_pre)
            
    elif enc_method=='Encoding Version 3':
        if len(n_cols)>0:
            scaler,n_cols=fit_StandardScaler(train_df,target)
            train_df[n_cols] = scaler.transform(train_df[n_cols])
            test_df[n_cols] = scaler.transform(test_df[n_cols])  
        if len(c_cols)>0:
            fit_pre=fit_Label_Encoding(train_df,target)  
            train_df=transform_Label_Encoding(train_df,fit_pre)
            test_df=transform_Label_Encoding(test_df,fit_pre)
            
    elif enc_method=='Encoding Version 4':
        if len(n_cols)>0:
            scaler,n_cols=fit_MinmaxScaler(train_df,target)
            train_df[n_cols] = scaler.transform(train_df[n_cols])
            test_df[n_cols] = scaler.transform(test_df[n_cols])   
        if len(c_cols)>0:
            fit_pre=fit_Label_Encoding(train_df,target)  
            train_df=transform_Label_Encoding(train_df,fit_pre)
            test_df=transform_Label_Encoding(test_df,fit_pre)

    if len(c_cols)==0:fit_pre=None
    if len(n_cols)==0:scaler=None
    
    if imp_method=='Simple':  
        imputer=fit_SimpleImp(train_df,target=target,strat='mean')
        
    elif imp_method=='KNN':  
        imputer=fit_KnnImp(train_df,target=target,neighbors=5)
        
    elif imp_method=='Iterative':
        imputer=fit_IterImp(train_df,target=target,order='ascending')

    elif imp_method=='Undefined':
        imputer=None

    Dataframe=train_df.reset_index(drop=True)
        
    ## Fit_Encoding_Version
    
    fit_atl={'enc_version':(enc_method,imp_method,target),
             'null_imputer':imputer,
             'cols':(list(Dataframe.columns),c_cols,n_cols),
             'scaler':scaler,
             'encod':fit_pre,
             }
    
    return fit_atl

def data_processing(Dataset:pd.DataFrame,
                    fit_atl:dict):
    
############################## Transformation Procediment ##############################

    df=Dataset.copy()
    
    df=engin_date(df)
    cols,c_cols,n_cols=fit_atl["cols"]
    enc_method,imp_method,target=fit_atl["enc_version"]
    scaler,input_cols=fit_atl["scaler"],c_cols+n_cols
    fit_pre=fit_atl["encod"]
    imputer=fit_atl["null_imputer"]
    
    df=df[cols]
    
    if len(n_cols)>0:
        df[n_cols] = scaler.transform(df[n_cols])
    
    if enc_method=='Encoding Version 1' or enc_method=='Encoding Version 2':
        if len(c_cols)>0:
            df=transform_IDF_Encoding(df,fit_pre)
    elif enc_method=='Encoding Version 3' or enc_method=='Encoding Version 4':
        if len(c_cols)>0:
            df=transform_Label_Encoding(df,fit_pre)
            
    if imp_method != "Undefined" and df[input_cols].isnull().sum().sum() != 0:
        if imp_method=='Simple':  
            df=transform_SimpleImp(df,target=target,imputer=imputer)
        elif imp_method=='KNN':  
            df=transform_KnnImp(df,target=target,imputer=imputer)
        elif imp_method=='Iterative':
            df=transform_IterImp(df,target=target,imputer=imputer)

    return df