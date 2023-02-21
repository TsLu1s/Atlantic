import numpy as np
import pandas as pd
import cane
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from .atl_feat_eng import num_cols, cat_cols, divide_dfs
from .atl_performance import pred_eval

def fit_SimpleImp(df:pd.DataFrame,
                   target:str,
                   strat:str='mean'):
    """
    The fit_SimpleImp function fits a SimpleImputer to the dataframe. 
    The function returns the fitted SimpleImputer object.
    
    :param df:pd.DataFrame: Specify the dataframe that you want to fit
    :param target:str: Specify the target variable
    :param strat:str: Specify the strategy to use when replacing missing values
    :return: A simpleimputer object
    """
    
    df_=df.copy()
    df=df.loc[:, df.columns != target]
    input_cols= list(df.columns)


    imputer = SimpleImputer(missing_values=np.nan, strategy=strat)
    imputer.fit(df)
    
    return imputer

def transform_SimpleImp(df:pd.DataFrame,
                         target:str,
                         imputer):
    """
    The transform_SimpleImp function takes a dataframe and imputes missing values using the SimpleImputer. 
    The function returns a new dataframe with the imputed values in place of NaN's.

    :param df:pd.DataFrame: Specify the dataframe that is to be transformed
    :param target:str: Specify the target column in the dataframe
    :param imputer: Select the imputer to be used
    :return: The dataframe with the imputed values in the columns that were transformed
    """
    
    df=df.reset_index(drop=True)  
    df_=df.copy()
    
    df=df.loc[:, df.columns != target]
    input_cols= list(df.columns)
    
    df = imputer.transform(df[input_cols])
    df = pd.DataFrame(df, columns = input_cols)

    df_[input_cols]=df[input_cols]
    
    return df_

def fit_KnnImp(df:pd.DataFrame, 
                target:str, 
                neighbors:int=5):
    
    df_=df.copy()
    df=df.loc[:, df.columns != target]
    imputer = KNNImputer(n_neighbors=neighbors)
    imputer.fit(df)
    
    return imputer

def transform_KnnImp(df:pd.DataFrame,
                      target:str,
                      imputer):
    
    df=df.reset_index(drop=True)
    df_=df.copy()
    
    df=df.loc[:, df.columns != target]
    df = pd.DataFrame(imputer.transform(df),columns = df.columns) 
    
    input_cols= list(df.columns)
    df_[input_cols]=df[input_cols]
    
    return df_

def fit_IterImp(df:pd.DataFrame, 
                 target:str, 
                 order:str='ascending'):

    df_=df.copy()
    df=df.loc[:, df.columns != target]
    imputer = IterativeImputer(imputation_order=order,max_iter=10,random_state=0,n_nearest_features=None)
    imputer=imputer.fit(df)

    return imputer

def transform_IterImp(df:pd.DataFrame,
                      target:str,
                      imputer):
    
    df=df.reset_index(drop=True)
    df_=df.copy()
    
    df=df.loc[:, df.columns != target]
    input_cols= list(df.columns)
    
    df = pd.DataFrame(imputer.transform(df))
    df.columns = input_cols
    
    df_[input_cols]=df[input_cols]
    
    return df_

def null_substitution_method(train:pd.DataFrame, 
                             test:pd.DataFrame, 
                             target:str, 
                             enc_method:str, 
                             pred_type:str,
                             eval_metric:str):
    """
    This function compares the performance of different imputation techniques
    on a dataset with missing values.
    
    Parameters:
    train : pd.DataFrame
        The training set
    test : pd.DataFrame
        The test set
    target : str
        The target variable
    enc_method : str
        The encoding method to be used
    pred_type : str
        The type of prediction: 'Reg' for regression or 'Class' for classification
    eval_metric : str
        The evaluation metric to be used
    """
    train_,test_=train.copy(),test.copy()
    sel_cols= list(train.columns)

    n_cols,c_cols=num_cols(train,target),cat_cols (train,target) 

    if pred_type=='Reg':
        metric='MAE'
    elif pred_type=='Class':
        metric='AUC'
    
    if enc_method=='Encoding Version 1':
        train, test=encoding_v1(train, test,target)
    elif enc_method=='Encoding Version 2':
        train, test=encoding_v2(train, test,target)
    elif enc_method=='Encoding Version 3':
        train, test=encoding_v3(train, test,target)
    elif enc_method=='Encoding Version 4':
        train, test=encoding_v4(train, test,target)
        
    
    print('Simple Imputation Loading')
    imputer=fit_SimpleImp(train,
                          target=target,
                          strat='mean')

    train_simple=transform_SimpleImp(train,
                                     target=target,
                                     imputer=imputer)
    
    test_simple=transform_SimpleImp(test,
                                    target=target,
                                    imputer=imputer)
    
    print('KNN Imputation Loading')
    imputer_knn=fit_KnnImp(train,
                           target=target,
                           neighbors=5)

    train_knn=transform_KnnImp(train,
                               target=target,
                               imputer=imputer_knn)

    test_knn=transform_KnnImp(test,
                              target=target,
                              imputer=imputer_knn)
    
    print('Iterative Imputation Loading')
    imputer_iter=fit_IterImp(train,
                             target=target,
                             order='ascending')

    train_iter=transform_IterImp(train,
                                 target=target,
                                 imputer=imputer_iter)

    test_iter=transform_IterImp(test,
                                target=target,
                                imputer=imputer_iter)

    simple_perf=pred_eval(train_simple, test_simple,target)
    knn_perf=pred_eval(train_knn, test_knn,target) 
    iter_perf=pred_eval(train_iter, test_iter,target)

    list_=[knn_perf,iter_perf,simple_perf]
    perf_imp=pd.concat(list_)
    perf_imp=perf_imp.reset_index()
    perf_imp = perf_imp.sort_values([eval_metric], ascending=True)
    
    mae_simple=simple_perf[eval_metric].sum()
    mae_knn=knn_perf[eval_metric].sum()
    mae_Iterartive=iter_perf[eval_metric].sum()
    
    print('Null Imputation Methods Performance:')
    
    print('KNN Performance: ', round(mae_knn, 4), '\n Iterative Performance: ', 
          round(mae_Iterartive, 4),'\n Simple Performance: ', round(mae_simple, 4))
    
    list_imp=[mae_Iterartive,mae_knn,mae_simple]
    
    list_imp.sort()
    imp_method=''
    
    if pred_type=='Class':
        list_imp.sort(reverse=True)

    if list_imp[0]==mae_Iterartive:
        imp_method='Iterative'
        train, test=train_iter.copy(),test_iter.copy()
        print('Iterative Imputation Algorithm was chosen with an ', metric, ' of: ', round(mae_Iterartive, 4))
    elif list_imp[0]==mae_knn:
        imp_method='KNN'
        train, test=train_knn.copy(), test_knn.copy()
        print('KNN Imputation Algorithm was chosen with an ', metric, ' of: ', round(mae_knn, 4))
    elif list_imp[0]==mae_simple:
        imp_method='Simple'  
        train, test=train_simple.copy(),test_simple.copy()
        print('Simple Imputation Algorithm was chosen with an ', metric, ' of: ', round(mae_simple, 4))

    return train, test,imp_method


def encoding_v1(train:pd.DataFrame, test:pd.DataFrame, target:str):
    
    """
    Encoding method version 1.
    This function applies two encoding techniques on the input dataframe:
        - Scale numerical variables using StandardScaler
        - Encode categorical variables using IDF
        
    Parameters:
        train (pd.DataFrame): The training dataset
        test (pd.DataFrame): The test dataset
        target (str): The target variable
    
    Returns:
        _train, _test : A tuple of the transformed training and test datasets.
    """

    _train,_test = train.copy(),test.copy()
    
    n_cols,c_cols=num_cols(_train,target),cat_cols(_train,target) 

    if len(n_cols)>0:
        scaler,n_cols= fit_StandardScaler(_train,target)
        _train[n_cols] = scaler.transform(_train[n_cols])
        _test[n_cols] = scaler.transform(_test[n_cols])    
    if len(c_cols)>0:
        idf_fit=fit_IDF_Encoding(_train,target)
        _train=transform_IDF_Encoding(_train,idf_fit)
        _test=transform_IDF_Encoding(_test,idf_fit)

    return _train,_test

def encoding_v2(train:pd.DataFrame, test:pd.DataFrame, target:str):

    _train,_test = train.copy(),test.copy()
    
    n_cols,c_cols=num_cols(_train,target),cat_cols(_train,target)

    if len(n_cols)>0:
        scaler,n_cols=fit_MinmaxScaler(_train,target)
        _train[n_cols] = scaler.transform(_train[n_cols])
        _test[n_cols] = scaler.transform(_test[n_cols])   
    if len(c_cols)>0:
        idf_fit=fit_IDF_Encoding(_train,target)
        _train=transform_IDF_Encoding(_train,idf_fit)
        _test=transform_IDF_Encoding(_test,idf_fit)

    return _train,_test

def encoding_v3(train:pd.DataFrame, test:pd.DataFrame, target:str):

    _train,_test = train.copy(),test.copy()
    
    n_cols,c_cols=num_cols(_train,target),cat_cols(_train,target) 
    
    if len(n_cols)>0:
        scaler,n_cols=fit_StandardScaler(_train,target)
        _train[n_cols] = scaler.transform(_train[n_cols])
        _test[n_cols] = scaler.transform(_test[n_cols])  
    if len(c_cols)>0:
        le_fit=fit_Label_Encoding(_train,target)  
        _train=transform_Label_Encoding(_train,le_fit)
        _test=transform_Label_Encoding(_test,le_fit)
    return _train,_test

def encoding_v4(train:pd.DataFrame, test:pd.DataFrame, target:str):

    _train,_test = train.copy(),test.copy()
    
    n_cols,c_cols=num_cols(_train,target),cat_cols(_train,target)

    if len(n_cols)>0:
        scaler,n_cols=fit_MinmaxScaler(_train,target)
        _train[n_cols] = scaler.transform(_train[n_cols])
        _test[n_cols] = scaler.transform(_test[n_cols])   
    if len(c_cols)>0:
        le_fit=fit_Label_Encoding(_train,target)  
        _train=transform_Label_Encoding(_train,le_fit)
        _test=transform_Label_Encoding(_test,le_fit)
    
    return _train,_test

######### MultiColumn LabelEncoding

def fit_Label_Encoding(Dataset:pd.DataFrame,target:str):
    """
    This function performs the Label Encoding for categorical variables in a dataset.
    Args:
        Dataset (pd.DataFrame): The dataset that you want to encode.
        target (str): The name of the target variable.
    Returns:
        le_dict (dict): A dictionary of label encoders for each categorical variable.
    """
    encoders=cat_cols(Dataset,target)
    df,list_cols,list_le=Dataset.copy(),[],[]
    
    for c in encoders:
        le = LabelEncoder()
        list_cols.append(c),list_le.append(le.fit(df[c]))
    le_dict = {list_cols[i]: list_le[i] for i in range(len(list_cols))}
    
    return le_dict

def transform_Label_Encoding(Dataset:pd.DataFrame,le_fit:dict):
    """
    This function receives a dataset and a pre-trained label encoding dict.
    It maps any unseen values in the dataset to '<unknown>', appends it to the classes_
    array of the label encoder, and then applies the encoding to the dataframe.
    """
    encoders=list(le_fit.keys())
    df=Dataset.copy()

    for c in encoders:
        le=le_fit[c]  
        df[c] = df[c].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        df[c] = le.transform(df[c])
        
    return df 

######### MultiColumn IDF_Encoding 

def fit_IDF_Encoding(Dataset:pd.DataFrame,target:str):
    df=Dataset.copy()
    
    encoders=cat_cols(df,target)
    IDF_filter = cane.idf(df, n_coresJob=2,disableLoadBar = False, columns_use = encoders)
    idf_fit = cane.idfDictionary(Original = df, Transformed = IDF_filter, columns_use = encoders)

    return idf_fit

def transform_IDF_Encoding(Dataset:pd.DataFrame,idf_fit:dict):
    
    encoders=list(idf_fit.keys())
    df=Dataset.copy()
        
    for col in encoders:
        df[col] = (df[col].map(idf_fit[col]).fillna(max(idf_fit[col].values())))
        
    return df 

######### MultiColumn OneHotEncoding

def fit_OneHot_Encoding(Dataset:pd.DataFrame,target:str,n_distinct:int=10):
        
    df,list_cols,list_le=Dataset.copy(),[],[]
    drop_org_cols,list_ohe=True,[] 
    encoders=cat_cols(df,target)

    if len(encoders)>0:
        for enc in encoders:
            if len(list(dict.fromkeys(df[enc].tolist())))<=n_distinct:  ## Less than n distinct elements in col
                ohe = OneHotEncoder(handle_unknown = 'ignore')
                
                list_cols.append(enc),list_le.append(ohe.fit(df[[enc]]))
    
    ohe_fit = {list_cols[i]: list_le[i] for i in range(len(list_cols))}
    ohe_fit["n_distinct"]=n_distinct
    
    return ohe_fit

def transform_OneHot_Encoding(Dataset:pd.DataFrame,ohe_fit:dict):
    
    df= Dataset.copy()
    drop_org_cols,list_ohe,n_distinct=True,[],ohe_fit["n_distinct"]
    del ohe_fit["n_distinct"]
    encoders=list(ohe_fit.keys())

    if len(encoders)>0:
        for enc in encoders:
            col_n=[]
            if len(list(dict.fromkeys(df[enc].tolist())))<n_distinct:  ## Less than n distinct elements in col
                list_ohe.append(enc)
                ohe = ohe_fit[enc] 
                x=ohe.transform(df[[enc]]).toarray()
                df_copy = pd.DataFrame(x)
                enc_cols = list(df_copy.columns)
                for element in range(len(enc_cols)):
                    name=enc+"_"+str(element+1)
                    col_n.append(name)
                df_copy = df_copy.rename(columns={enc_cols[i]: col_n[i] for i in range(len(enc_cols))})

                df = pd.concat([df, df_copy.set_index(df.index)], axis=1)
    df,ohe_fit['n_distinct']=df.drop(list_ohe, axis=1),n_distinct
    
    return df

############################# Scalers

def fit_StandardScaler(Dataset:pd.DataFrame,target:str):
    
    df,n_cols=Dataset.copy(),num_cols(Dataset,target)
    
    scaler = StandardScaler()
    scaler = scaler.fit(df[n_cols])

    return scaler, n_cols

def fit_MinmaxScaler(Dataset:pd.DataFrame,target:str):
    
    df,n_cols=Dataset.copy(),num_cols(Dataset,target)
    
    scaler = MinMaxScaler() 
    scaler = scaler.fit(df[n_cols])

    return scaler, n_cols

def fit_RobustScaler(Dataset:pd.DataFrame,target:str):
    
    df,n_cols=Dataset.copy(),num_cols(Dataset,target)
    
    scaler = RobustScaler()
    scaler = scaler.fit(df[n_cols])
    
    return scaler, n_cols


#######################  Select Best Encoding Method  ######################


def Select_Encoding_Method(train:pd.DataFrame, 
                           test:pd.DataFrame, 
                           target:str, 
                           pred_type:str,
                           eval_metric:str):
    """
    The function Select_Encoding_Method() is used for selecting the best encoding method for a given dataset, it takes in 4 arguments:

    train: a dataframe containing the training data
    test: a dataframe containing the test data
    target: a string representing the target column
    pred_type: a string indicating the type of prediction. It can either be 'Reg' for regression or 'Class' for classification
    eval_metric: a string indicating the metric used to evaluate the performance of the predictions.
    First, it copies the input datasets and create some variables to store the results of the encoding methods.
    Then the function applies 4 different encoding methods (encoding_v1(), encoding_v2(), encoding_v3(), and encoding_v4()) to both the train and test datasets.
    The function then evaluates the performance of these 4 encoded datasets using the pred_eval() function and a metric defined by the input eval_metric.
    It will print the performance of each method and depending on pred_type it will select the best encoding method based on the chosen metric.
    """
    train_,test_=train.copy(),test.copy()
    _train_,_test_=train_.copy(),test_.copy()


    train_v1, test_v1=encoding_v1(_train_, _test_,target)
    if (train_v1.isnull().sum().sum() or test_v1.isnull().sum().sum()) != 0:
        imputer_v1=fit_SimpleImp(train_v1,target=target)
        train_v1=transform_SimpleImp(train_v1,target=target,imputer=imputer_v1)
        test_v1=transform_SimpleImp(test_v1,target=target,imputer=imputer_v1)
        
    train_v2, test_v2=encoding_v2(_train_, _test_,target)
    if (train_v2.isnull().sum().sum() or test_v2.isnull().sum().sum()) != 0:
        imputer_v2=fit_SimpleImp(train_v2,target=target)
        train_v2=transform_SimpleImp(train_v2,target=target,imputer=imputer_v2)
        test_v2=transform_SimpleImp(test_v2,target=target,imputer=imputer_v2)
        
    train_v3, test_v3=encoding_v3(_train_, _test_,target)
    if (train_v3.isnull().sum().sum() or test_v3.isnull().sum().sum()) != 0:
        imputer_v3=fit_SimpleImp(train_v3,target=target)
        train_v3=transform_SimpleImp(train_v3,target=target,imputer=imputer_v3)
        test_v3=transform_SimpleImp(test_v3,target=target,imputer=imputer_v3)
        
    train_v4, test_v4=encoding_v4(_train_, _test_,target)
    if (train_v4.isnull().sum().sum() or test_v4.isnull().sum().sum()) != 0:
         imputer_v4=fit_SimpleImp(train_v4,target=target)
         train_v4=transform_SimpleImp(train_v4,target=target,imputer=imputer_v4)
         test_v4=transform_SimpleImp(test_v4,target=target,imputer=imputer_v4)

    pred_perf_v1=pred_eval(train_v1, test_v1,target)
    pred_perf_v2=pred_eval(train_v2, test_v2,target)
    pred_perf_v3=pred_eval(train_v3, test_v3,target)
    pred_perf_v4=pred_eval(train_v4, test_v4,target)
    
    p_v1=pred_perf_v1[eval_metric].sum()
    p_v2=pred_perf_v2[eval_metric].sum()
    p_v3=pred_perf_v3[eval_metric].sum()
    p_v4=pred_perf_v4[eval_metric].sum()
    
    if pred_type=='Reg':
        print(' ')
        print('Predictive Performance Encoding Versions:')
        print('\n MAE Version 1 [IDF + StandardScaler] : ', round(p_v1, 4),
              '\n MAE Version 2 [IDF + MinMaxScaler] : ', round(p_v2, 4),
              '\n MAE Version 3 [Label + StandardScaler] : ', round(p_v3, 4),
              '\n MAE Version 4 [Label + MinMaxScaler] : ', round(p_v4, 4))
        metric='MAE'
    elif pred_type=='Class':
        print('Predictive Performance Encoding Versions:')
        print('\n AUC Version 1 [IDF + StandardScaler] : ', round(p_v1, 4),
              '\n AUC Version 2 [IDF + MinMaxScaler] : ', round(p_v2, 4),
              '\n AUC Version 3 [Label + StandardScaler] : ', round(p_v3, 4),
              '\n AUC Version 4 [Label + MinMaxScaler] : ', round(p_v4, 4))
        metric='AUC'
    
    list_encoding=[p_v1,p_v2,p_v3,p_v4]
    list_encoding.sort()
    enc_method=''
    
    if pred_type=='Class':
        list_encoding.sort(reverse=True)
        
    if list_encoding[0]==p_v1:
        enc_method='Encoding Version 1'
        print('Encoding Version 1 was choosen with an ', metric, ' of: ', round(p_v1, 4))
        
    elif list_encoding[0]==p_v2:
        enc_method='Encoding Version 2'
        print('Encoding Version 2 was choosen with an ', metric, ' of: ', round(p_v2, 4))    
    
    elif list_encoding[0]==p_v3:
        enc_method='Encoding Version 3'
        print('Encoding Version 3 was choosen with an ', metric, ' of: ', round(p_v3, 4))
    
    elif list_encoding[0]==p_v4:
        enc_method='Encoding Version 4'
        print('Encoding Version 4 was choosen with an ', metric, ' of: ', round(p_v4, 4))
    
    return enc_method
