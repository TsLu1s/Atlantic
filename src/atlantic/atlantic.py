import numpy as np
import pandas as pd
import cane
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import h2o
from h2o.automl import H2OAutoML
from statsmodels.stats.outliers_influence import variance_inflation_factor

h2o.init()

##################################################### Loading/Split do Dataset #############################################################

def reset_index_DF(Dataset:pd.DataFrame):
    
    Dataset=Dataset.reset_index()
    Dataset.drop(Dataset.columns[0], axis=1, inplace=True)
    
    return Dataset

def split_dataset(Dataset:pd.DataFrame, Split_Racio:float):
    
    assert Split_Racio>=0.5 and Split_Racio<=0.95 , 'Split_Racio value should be in [0.5,0.95[ interval'
    
    train, test= train_test_split(Dataset, train_size=Split_Racio)

    return train,test

def transform_dataset(Dataset:pd.DataFrame, Dataframe:pd.DataFrame):

    df_ = Dataset.copy()
    _df = Dataframe.copy()

    for col in _df:
        df_[[col]]=_df[[col]]
        
    return df_

def target_type(Dataset:pd.DataFrame, target:str):  
    
    df=Dataset[[target]]
    reg_target,class_target=df.select_dtypes(include=['int','float']).columns.tolist(),df.select_dtypes(include=['object']).columns.tolist()
    if len(class_target)==1:
        pred_type='Class'
        eval_metric='Accuracy'
    elif len(reg_target)==1:
        pred_type='Reg'
        eval_metric='Mean Absolute Error'
        
    return pred_type, eval_metric

############################################################# Datetime Feature Engineering ######################################################

def slice_timestamp(Dataset:pd.DataFrame,date_col:str='Date'):
    
    df=Dataset.copy()
    cols=list(df.columns)
    for col in cols:
        if col==date_col:
            df[date_col] = df[date_col].astype(str)
            df[date_col] = df[date_col].str.slice(0,19)
            df[date_col] = pd.to_datetime(df[date_col])
            
    return df


def engin_date(Dataset:pd.DataFrame, drop:bool=True):
    
    """
    The engin_date function takes a DataFrame and returns a DataFrame with the date features engineered.
    The function has two parameters: 
    Dataset: A Pandas DataFrame containing at least one column of datetime data. 
    drop: A Boolean value indicating whether or not to drop the original datetime columns from the returned dataset.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param drop:bool=False: Decide whether or not to drop the original datetime columns from the returned dataset
    :return: A dataframe with the date features engineered
    """

    Dataset_=Dataset.copy()
    Df=Dataset_.copy()
    Df=slice_timestamp(Df)
    
    x=pd.DataFrame(Df.dtypes)
    x['column'] = x.index
    x=x.reset_index().drop(['index'], axis=1).rename(columns={0: 'dtype'})
    a=x.loc[x['dtype'] == 'datetime64[ns]']

    list_date_columns=[]
    for col in a['column']:
        list_date_columns.append(col)

    def create_date_features(df,elemento):
        
        df[elemento + '_day_of_month'] = df[elemento].dt.day
        df[elemento + '_day_of_week'] = df[elemento].dt.dayofweek + 1
        df[[elemento + '_is_wknd']] = df[[elemento + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7], 
                            [0, 0, 0, 0, 0, 1, 1 ]) 
        df[elemento + '_month'] = df[elemento].dt.month
        df[elemento + '_day_of_year'] = df[elemento].dt.dayofyear
        df[elemento + '_year'] = df[elemento].dt.year
        df[elemento + '_hour']=df[elemento].dt.hour
        df[elemento + '_minute']=df[elemento].dt.minute
        df[elemento + '_Season']=''
        winter = list(range(1,80)) + list(range(355,370))
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        df.loc[(df[elemento + '_day_of_year'].isin(spring)), elemento + '_Season'] = '2'
        df.loc[(df[elemento + '_day_of_year'].isin(summer)), elemento + '_Season'] = '3'
        df.loc[(df[elemento + '_day_of_year'].isin(fall)), elemento + '_Season'] = '4'
        df.loc[(df[elemento + '_day_of_year'].isin(winter)), elemento + '_Season'] = '1'
        df[elemento + '_Season']=df[elemento + '_Season'].astype(np.int64)
        
        return df 
    
    if drop==True:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
            Df=Df.drop(elemento,axis=1)
    elif drop==False:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
    #if len(list_date_columns)>=1:
    #    print('Date Time Feature Generation')
        
    return Df

############################################################# Enconding Lists ###################################################################

def num_cols(Dataset:pd.DataFrame, target:str):
    
    n_cols=Dataset.select_dtypes(include=['int','float']).columns.tolist()
    
    for col in n_cols:
        if col==target:
            n_cols.remove(target)
            
    return n_cols

def cat_cols(Dataset:pd.DataFrame, target):

    c_cols=Dataset.select_dtypes(include=['object']).columns.tolist()

    for col in c_cols:
        if col==target:
            c_cols.remove(target)
            
    return c_cols 

############################################################# Nulls Treatment ###################################################################

def del_nulls_target(Dataset:pd.DataFrame, target:str):
        
    Dataset=Dataset[Dataset[target].isnull()==False]
    
    return Dataset

def remove_columns_by_nulls(Dataset:pd.DataFrame, percentage:int): ## Colunas 
    
    assert percentage>0 and percentage<=100 , 'percentage should not exceed value of 100'
    df=Dataset.copy()
    perc = percentage
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    df = df.dropna( axis=1,
                thresh=min_count)
    df = df.loc[:, (df==0).mean()*100 < perc]
    df = df.loc[:, ~(df.apply(pd.Series.value_counts, normalize=True).max() > perc/100)]

    return df

############################################################# Null Imputation ########################################################################################

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
    
    #df=reset_index_DF(df)
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
    
    df=reset_index_DF(df)    
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
    
    df=reset_index_DF(df)   
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
    
    df=reset_index_DF(df)   
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

    List=[knn_perf,iter_perf,simple_perf]
    perf_imp=pd.concat(List)
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

############################################################# Encoding Pipeline Methods ####################################################

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

######################################################## Encoding Updated Methods - >0.0.9 Version ###########################################

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
    idf_fit = cane.idfDictionary(Original = df, Transformed = IDF_filter, columns_use = encoders) #, targetColumn=target)

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
    print(encoders)
    if len(encoders)>0:
        for enc in encoders:
            if len(list(dict.fromkeys(df[enc].tolist())))<=n_distinct:  ## Less/= than n distinct elements in col
                ohe = OneHotEncoder(handle_unknown = 'ignore')
                print("******", enc)              
                
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

########################################################### Feature Selection ##############################################################
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
    Selected_Cols: list
            list of selected features
    Selected_Importance: pandas DataFrame, shape = (n_features, 2)
            Dataframe with the relative importance and variables
    """
    assert total_vi>=0.5 and total_vi<=1 , 'total_vi value should be in [0.5,1] interval'
    assert h2o_fs_models>=1 and h2o_fs_models<=50 , 'h2o_fs_models value should be in [0,50] interval'
    
    train_=Dataset.copy()
    train=train_.copy()
    
    if encoding_fs==True:
        le =LabelEncoder()
        cols=cat_cols(train_,target)   
        train_=train_[cols]
        train_ = train_.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
        train=transform_dataset(train,train_)
    elif encoding_fs==False:
        print('    Encoding method was not applied    ')

    train=remove_columns_by_nulls(train, 99) 
    
    Input_Cols= list(train.columns)
    Input_Cols.remove(target)
        
    train_h2o=h2o.H2OFrame(train)
    
    pred_type, eval_metric=target_type(train, target)
    
    if pred_type=='Class': train_h2o[target] = train_h2o[target].asfactor()
    
    aml = H2OAutoML(max_models=h2o_fs_models,nfolds=3 , seed=1, exclude_algos = ['GLM', 'DeepLearning', 'StackedEnsemble'],sort_metric = 'AUTO')
    aml.train(x=Input_Cols,y=target,training_frame=train_h2o)
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
    Selected_Cols=[]
    for rows in va_imp_df['variable']:
        Selected_Cols.append(rows)   
    print('Total amount of selected input columns: ', len(Selected_Cols))
    print('Total relative importance percentage of the selected columns: ', round(sum_va_imp*100, 4), '%')
    if len(Selected_Cols)>=5:
        list_t5_cols=Selected_Cols[0:5]
        print('Top 5 Most Important Input Columns: ', list_t5_cols)
    Selected_Cols.append(target)
    Selected_Importance=va_imp_df.copy()    

    return Selected_Cols, Selected_Importance

###############################################  VIF #########################################

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
    
    Input_Cols= list(Dataset.columns)
    Input_Cols.remove(target)
    Dataset_=Dataset[Input_Cols]
    vif_df=calc_vif(Dataset_)
    sel_cols=Input_Cols
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
        Cols_vif,vif_df=feature_selection_vif(_train__,target,vif_ratio)
        print('Number of Selected VIF Columns: ', len(Cols_vif), 
              '\n Removed Columns with VIF (Feature Selection - VIF):', len(sel_cols) - len(Cols_vif), 
              '\n Selected Columns:', Cols_vif)
        _train__=_train__[Cols_vif]
        _test__=_test__[Cols_vif]
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
            _train_=_train_[Cols_vif]
            _test_=_test_[Cols_vif]
        else:
            print('The VIF filtering method was not applied    ')
    elif pred_type=='Class':
        if p_default_vif>default_p:
            print('The VIF filtering method was applied    ')
            _train_=_train_[Cols_vif]
            _test_=_test_[Cols_vif]
        else:
            print('The VIF filtering method was not applied    ')
    return _train_, _test_
   
########################################################### Metrics ########################################################################

def metrics_regression(y_real, y_pred): 
    """
    Function to calculate various regression model evaluation metrics
    Parameters:
    y_real: array-like, shape = (n_samples)
            Real values of the target
    y_pred: array-like, shape = (n_samples)
            Predicted values of the target
    Returns:
    Metrics_Prev_Regression: dictionary with keys as the metrics names, and values as the respective values
    """
    mae=mean_absolute_error(y_real, y_pred)
    mape= mean_absolute_percentage_error(y_real, y_pred)
    mse=mean_squared_error(y_real, y_pred)
    evs= explained_variance_score(y_real, y_pred)
    maximo_error= max_error(y_real, y_pred)
    r2=r2_score(y_real, y_pred)
    Metrics_Prev_Regression= {'Mean Absolute Error': mae, 
                              'Mean Absolute Percentage Error': mape,
                              'Mean Squared Error': mse,
                              'Explained Variance Score': evs, 
                              'Max Error': maximo_error,
                              'R2 Score':r2}  
    
    return Metrics_Prev_Regression

def metrics_classification(y_true, y_pred):

    accuracy_metric = accuracy_score(y_true, y_pred)
    precision_metric = precision_score(y_true, y_pred,average='micro')
    f1_macro_metric = f1_score(y_true, y_pred,average='macro')
    
    Metrics_Pred_Classification= {
                              'Accuracy': accuracy_metric,
                              'Precision Micro': precision_metric,
                              'F1 Score Macro':f1_macro_metric,
                              }
    
    return Metrics_Pred_Classification

def metrics_binary_classification(y_true, y_pred):
    
    f1_metric=f1_score(y_true, y_pred)
    accuracy_metric = accuracy_score(y_true, y_pred)
    precision_metric = precision_score(y_true, y_pred)
    recall_metric = recall_score(y_true, y_pred)
    average_precision_metric = average_precision(y_true, y_pred)
    balanced_accuracy_metric = balanced_accuracy(y_true, y_pred)
    
    Metrics_Pred_Classification= {'Accuracy': accuracy_metric, 
                              'Precision Score': precision_metric,
                              'F1 Score': f1_metric,
                              'Recall Score': recall_metric,
                              'Average Precision': average_precision_metric, 
                              'Balanced Accuracy': balanced_accuracy_metric
                              }

    return Metrics_Pred_Classification

###########################################################  Tree Based Algorithms   #######################################################

def divide_dfs(train:pd.DataFrame,test:pd.DataFrame,target:str):
    
    sel_cols=list(train.columns)
    sel_cols.remove(target)
    sel_cols.append(target) 
    train,test=train[sel_cols],test[sel_cols]  
    
    X_train = train.iloc[:, 0:(len(sel_cols)-1)].values
    X_test = test.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = train.iloc[:, (len(sel_cols)-1)].values
    y_test = test.iloc[:, (len(sel_cols)-1)].values
    
    return X_train,X_test,y_train,y_test


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
    Metrics_Final=(y+z)/2
    
    return Metrics_Final

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
    
    Pred_type, Eval_metric=target_type(Dataset_, target) ## Prediction Contextualization
    
############################## Feature Engineering Date Column ##############################

    train=engin_date(train)
    test=engin_date(test)

############################## Feature Selection ##############################
    
    sel_cols, Sel_Imp =feature_selection_h2o(train,target,total_vi,h2o_fs_models,encoding_fs)
    print('Selected Columns:', sel_cols)

    train=train[sel_cols]
    test=test[sel_cols]

############################## Encoding Method Selection ##############################   

    enc_method=Select_Encoding_Method(train,test,target,Pred_type,Eval_metric)

############################## NULL SUBSTITUTION + ENCODING APLICATION ##############################

    if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
        train, test,imp_method=null_substitution_method(train,test,target,enc_method,Pred_type,Eval_metric)
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

    Dataframe=reset_index_DF(train_df)
        
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
