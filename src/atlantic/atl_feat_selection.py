import numpy as np
import pandas as pd
import cane
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import h2o
from h2o.automl import H2OAutoML
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .atl_feat_eng import target_type, remove_columns_by_nulls
from .atl_performance import pred_eval
from .atl_processing import fit_Label_Encoding, transform_Label_Encoding

#h2o.init()

###################################  H2O Feature Selection ######################################

def feature_selection_h2o(Dataset:pd.DataFrame, target:str, total_vi :float=0.98, h2o_fs_models:int =7, encoding_fs:bool=True):
    """
    Function to select features using h2o's Autodml feature.
    Parameters:
    Dataset: pandas DataFrame, shape = (n_samples,n_features)
            Dataframe 
    target: str
            target variable
    total_vi : float
            total relative importance percentage of the selected columns, default is 0.98
    h2o_fs_models: int
            h2o Autodml feature models,default is 7
    encoding_fs : bool
            if encoding should be applied or not to the dataset, default is True
    Returns:
    selected_cols: list
            list of selected features
    selected_importance: pandas DataFrame, shape = (n_features, 2)
            Dataframe with the relative importance and variables
    """
    assert total_vi>=0.5 and total_vi<=1 , 'total_vi value should be in [0.5,1] interval'
    assert h2o_fs_models>=1 and h2o_fs_models<=50 , 'h2o_fs_models value should be in [0,50] interval'
    
    train_=Dataset.copy()
    train=train_.copy()
    
    if encoding_fs==True:
        le_fit=fit_Label_Encoding(train_,target)   
        train_=transform_Label_Encoding(train_,le_fit)
        train=train_.copy()
        
    elif encoding_fs==False:
        print('    Encoding method was not applied    ')

    train=remove_columns_by_nulls(train, 99) 
    
    input_cols= list(train.columns)
    input_cols.remove(target)
        
    train_h2o=h2o.H2OFrame(train)
    
    pred_type, eval_metric=target_type(train, target)
    
    if pred_type=='Class': train_h2o[target] = train_h2o[target].asfactor()
    
    aml = H2OAutoML(max_models=h2o_fs_models,nfolds=3 , seed=1, exclude_algos = ['GLM', 'DeepLearning', 'StackedEnsemble'],sort_metric = 'AUTO')
    aml.train(x=input_cols,y=target,training_frame=train_h2o)
    leaderboards = aml.leaderboard
    leaderboards_df = leaderboards.as_data_frame()
    print(leaderboards_df)
    
    list_id_model=[]
    
    for row in leaderboards_df['model_id']:  
        list_id_model.append(row)
     
    print('Selected Leaderboard Model: ', list_id_model[0])
        
    m = h2o.get_model(list_id_model[0])
    total_imp=m.varimp(use_pandas=True)
        
    va_imp=total_imp.copy()
        
    n=0.015
    va_imp_df = va_imp[va_imp['percentage'] > n]
    sum_va_imp=va_imp_df['percentage'].sum()
    for iteration in range(0,10):
        print('Approximate minimum value of Relative Percentage:',n)
            
        if sum_va_imp<=total_vi:
            va_imp_df = va_imp[va_imp['percentage'] > n]
            n=n*0.5
            sum_va_imp=va_imp_df['percentage'].sum()
        elif sum_va_imp>total_vi:
            break
    selected_cols=[]
    for rows in va_imp_df['variable']:
        selected_cols.append(rows)   
    print('Total amount of selected input columns: ', len(selected_cols))
    print('Total relative importance percentage of the selected columns: ', round(sum_va_imp*100, 4), '%')
    if len(selected_cols)>=5:
        list_t5_cols=selected_cols[0:5]
        print('Top 5 Most Important Input Columns: ', list_t5_cols)
    selected_cols.append(target)
    selected_importance=va_imp_df.copy()    

    return selected_cols, selected_importance

############################################### VIF ##########################################

def calc_vif(X):
    '''
    The calc_vif function calculates the variance inflation factor for each variable in a dataframe.
    It returns a pandas DataFrame with two columns: variables and VIF. The variables column contains 
    the names of the independent variables, while the VIF column contains their respective values.
    '''
    # Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.sort_values(['VIF'], ascending=False)
    return vif

def feature_selection_vif(Dataset:pd.DataFrame, target:str, VIF:float=10.0):
    """
    Function to select features based on VIF 
    Parameters:
    Dataset: pandas DataFrame, shape = (n_samples,n_features)
            Dataframe 
    target: str
            target variable
    VIF: float
            vif threshold ratio, default is 10.0
    Returns:
    sel_cols: list
            list of selected features
    vif_df : pandas DataFrame, shape = (n_features, 2)
            Dataframe with the vif and variables
    """   
    assert VIF>=3 and VIF<=30 , 'VIF value should be in [3,30] interval'
    
    input_cols= list(Dataset.columns)
    input_cols.remove(target)
    Dataset_=Dataset[input_cols]
    vif_df=calc_vif(Dataset_)
    sel_cols=input_cols
    for line in range(0,len(vif_df['VIF'])):
        if vif_df['VIF'].loc[vif_df['VIF'].idxmax()]>=VIF:
            vif_df.drop(vif_df['variables'].loc[vif_df['VIF']==vif_df['VIF'].max()].index, inplace=True)
            sel_cols=[]
            for rows in vif_df['variables']:
                sel_cols.append(rows)
        Dataset_=Dataset_[sel_cols]
        vif_df=calc_vif(Dataset_)
    sel_cols.append(target)

    return sel_cols,vif_df

def vif_performance_selection(train:pd.DataFrame, 
                              test:pd.DataFrame, 
                              target:str, 
                              vif_ratio:float=10.0):
    """
    Function to select features based on VIF performance
    Parameters:
    train: pandas DataFrame, shape = (n_samples,n_features)
            Training dataframe 
    test: pandas DataFrame, shape = (n_samples,n_features)
            Test dataframe 
    target: str
            target variable
    vif_ratio: float
            vif threshold ratio, default is 10.0
    Returns:
    _train_: pandas DataFrame, shape = (n_samples,n_features)
            Training dataframe with selected features
    _test_: pandas DataFrame, shape = (n_samples,n_features)
            Test dataframe with selected features
    """
    assert vif_ratio>=3 and vif_ratio<=30 , 'vif_ratio value should be in [3,30] interval'
    
    train_,test_=train.copy(),test.copy()
    _train_,_test_=train_.copy(),test_.copy()
    
    pred_type, eval_metric=target_type(_train_, target)
    
    if pred_type=='Class':
        metric='AUC'
    elif pred_type=='Reg':
        metric='MAE'
    
    sel_cols=list(train_.columns)
    
    perf_default = pred_eval(_train_, _test_,target)
    default_p=perf_default[eval_metric][0]
    
    _train__,_test__=train_.copy(),test_.copy()

    try:
        cols_vif,vif_df=feature_selection_vif(_train__,target,vif_ratio)
        print('Number of Selected VIF Columns: ', len(cols_vif), 
              '\n Removed Columns with VIF (Feature Selection - VIF):', len(sel_cols) - len(cols_vif), 
              '\n Selected Columns:', cols_vif)
        _train__=_train__[cols_vif]
        _test__=_test__[cols_vif]
    except Exception:
        print('traceback.format_exc: ', traceback.format_exc())
        
    pred_vif = pred_eval(_train__, _test__,target)
    p_default_vif=pred_vif[eval_metric][0]
    print('   ')
    print('Default Performance:',round(default_p, 4))
    print('Performance Default VIF:',round(p_default_vif, 4))
    if pred_type=='Reg':
        if p_default_vif<default_p:
            print('The VIF filtering method was applied    ')
            _train_=_train_[cols_vif]
            _test_=_test_[cols_vif]
        else:
            print('The VIF filtering method was not applied    ')
    elif pred_type=='Class':
        if p_default_vif>default_p:
            print('The VIF filtering method was applied    ')
            _train_=_train_[cols_vif]
            _test_=_test_[cols_vif]
        else:
            print('The VIF filtering method was not applied    ')
    return _train_, _test_
