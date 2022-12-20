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
    
    '''
    The transform_dataset function takes two datasets, one to be transformed and another that contains the 
    transformations. The transform_dataset function then transforms the dataset with all of its transformations.


    :param Dataset:pd.DataFrame: Pass the dataset to be transformed
    :param Dataset_Transf:pd.DataFrame: Transform the dataset:pd
    :return: A dataset with the transformed columns
    '''

    df_ = Dataset.copy()
    _df = Dataframe.copy()

    for col in _df:
        df_[[col]]=_df[[col]]
        
    return df_


def reindex_columns(Dataset:pd.DataFrame, Feature_Importance:list):
    
    '''
    The reindex_columns function takes a dataframe and a list of column names as input. 
    It returns the same dataframe with the columns in the order specified by the list of column names.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be reindexed
    :param Feature_Importance:list: Reindex the columns in the dataframe
    :return: A dataframe with the columns reordered to match the feature importance list
    '''
    
    total_cols=list(Dataset.columns)
    y=feature_importance+total_cols
    z=list(dict.fromkeys(y))
    Dataset=Dataset[z]
    
    return Dataset

def target_type(Dataset:pd.DataFrame, target:str):  
    
    '''
    The target_type function takes in a pandas dataframe and returns the type of prediction problem. 
    
    :param Dataset:pd.DataFrame: Pass the dataframe that we want to use for our analysis
    :param target:str: Identify the target variable
    '''

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
    """
    The slice_timestamp function takes a dataframe and returns the same dataframe with the date column sliced to just include
    the year, month, day and hour. This is done by converting all of the values in that column to strings then slicing them 
    accordingly. The function then converts those slices back into datetime objects so they can be used for further analysis.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be sliced
    :param date_col:str='Date': Specify the name of the column that contains the date information
    :return: A dataframe with the timestamp column sliced to only include the year, month and day
    """
    df=Dataset.copy()
    cols=list(df.columns)
    for col in cols:
        if col==date_col:
            df[date_col] = df[date_col].astype(str)
            df[date_col] = df[date_col].str.slice(0,19)
            df[date_col] = pd.to_datetime(df[date_col])
    return df


def engin_date(Dataset:pd.DataFrame, Drop:bool=True):
    
    '''
    The engin_date function takes a DataFrame and returns a DataFrame with the date features engineered.
    The function has two parameters: 
    Dataset: A Pandas DataFrame containing at least one column of datetime data. 
    Drop: A Boolean value indicating whether or not to drop the original datetime columns from the returned dataset.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param Drop:bool=False: Decide whether or not to drop the original datetime columns from the returned dataset
    :return: A dataframe with the date features engineered
    '''

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

    def create_date_features(df,element):
        
        df[element + '_day_of_month'] = df[element].dt.day
        df[element + '_day_of_week'] = df[element].dt.dayofweek + 1
        df[[element + '_is_wknd']] = df[[element + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7], 
                            [0, 0, 0, 0, 0, 1, 1 ]) 
        df[element + '_month'] = df[element].dt.month
        df[element + '_day_of_year'] = df[element].dt.dayofyear
        df[element + '_year'] = df[element].dt.year
        df[element + '_hour']=df[element].dt.hour
        df[element + '_minute']=df[element].dt.minute
        df[element + '_Season']=''
        winter = list(range(1,80)) + list(range(355,370))
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        df.loc[(df[element + '_day_of_year'].isin(spring)), element + '_Season'] = '2'
        df.loc[(df[element + '_day_of_year'].isin(summer)), element + '_Season'] = '3'
        df.loc[(df[element + '_day_of_year'].isin(fall)), element + '_Season'] = '4'
        df.loc[(df[element + '_day_of_year'].isin(winter)), element + '_Season'] = '1'
        df[element + '_Season']=df[element + '_Season'].astype(np.int64)
        
        return df 
    
    if Drop==True:
        for element in list_date_columns:
            Df=create_date_features(Df,element)
            Df=Df.drop(element,axis=1)
    elif Drop==False:
        for element in list_date_columns:
            Df=create_date_features(Df,element)
    #if len(list_date_columns)>=1:
    #    print('Date Time Feature Generation')
        
    return Df

############################################################# Enconding Lists ###################################################################

def numerical_columns(Dataset:pd.DataFrame, target:str):
    
    '''
    The numerical_columns function returns a list of the numerical columns in the dataframe.
    It takes two arguments: Dataset and target. 
    Dataset is the name of your dataset, and target is the name of your target variable
    
    :param Dataset:pd.DataFrame: Pass the dataframe that will be used to analyze
    :param target:str: Identify the target variable
    :return: A list with the numerical columns of a dataframe
    '''
    
    num_cols=Dataset.select_dtypes(include=['int','float']).columns.tolist()
    
    for col in num_cols:
        if col==target:
            num_cols.remove(target)
            
    return num_cols

def categorical_columns(Dataset:pd.DataFrame, target):
    
    '''
    The categorical_columns function returns a list of the categorical columns in the dataset.
    The function takes two arguments: Dataset and target. 
    Dataset is a pandas dataframe, and target is the name of your target variable column as a string.
    
    :param Dataset:pd.DataFrame: Specify the dataframe where we want to find the categorical columns
    :param target: Remove the target column from the list of categorical columns
    :return: A list with the categorical columns of a dataframe
    '''
    
    cat_cols=Dataset.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        if col==target:
            cat_cols.remove(target)
            
    return cat_cols 

############################################################# Nulls Treatment ###################################################################

def del_nulls_target(Dataset:pd.DataFrame, target:str):
        
    Dataset=Dataset[Dataset[target].isnull()==False]
    
    return Dataset

def remove_columns_by_nulls(Dataset:pd.DataFrame, percentage:int): ## Colunas 
    
    '''
    The remove_columns_by_nulls function removes columns from a dataframe that have more than the percentage of null values specified by the user.
    The function takes two arguments:
    Dataset - The dataset to be modified. This should be a pandas DataFrame object.
    percentage - A number between 0 and 100, inclusive, which represents the maximum allowable percentage of null values in any given column before it is removed from the dataframe.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be cleaned
    :param percentage:int: Specify the percentage of null values that a column must have in order to be removed
    '''
    
    assert percentage>0 and percentage<=100 , 'percentage should not exceed value of 100'
    df=Dataset.copy()
    perc = percentage
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    df = df.dropna( axis=1,
                thresh=min_count)
    df = df.loc[:, (df==0).mean()*100 < perc]
    df = df.loc[:, ~(df.apply(pd.Series.value_counts, normalize=True).max() > perc/100)]

    return df

############################################################# Null_Substitution ########################################################################################

def const_null_imputation(train:pd.DataFrame,
                          test:pd.DataFrame, 
                          target:str, 
                          imp_value:int=0):
    
    '''
    The const_null_imputation function imputes a constant value to all the null values in the dataframe.
    The function takes three arguments: train, test and target. 
    train is a pandas dataframe that contains training data with null values. 
    test is a pandas dataframe that contains test/validation/holdout set with null values. 
    target is the target variable name in both train and test sets.
    
    :param train:pd.DataFrame: Specify the training dataset
    :param test:pd.DataFrame: test the model on a different dataset
    :param target:str: Specify the target variable
    :param imp_value:int=0: Set the imputation value for all the null values in both train and test dataframes
    :return: The training and test dataframes with the null values imputed by a constant value
    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_=train.copy(),test.copy()
    
    train,test=train.loc[:, train.columns != target],test.loc[:, test.columns != target]
    train,test=train.fillna(imp_value),test.fillna(imp_value)
    train[target],test[target]=train_[target],test_[target]
    
    return train,test

def simple_null_imputation(train:pd.DataFrame,
                           test:pd.DataFrame,
                           target:str,
                           strat:str='mean'):
    
    '''
    The simple_null_imputation function takes in a train and test, as well as the target column name. 
    It then removes the target column from both train and test, imputes missing values with mean for numerical columns 
    and most frequent value for categorical columns. It returns two dataframes: one with train data and one with test data.
    
    :param train:pd.DataFrame: Specify the training dataframe
    :param test:pd.DataFrame: Impute the missing values in the test
    :param target:str: Specify the target variable for which we are imputing missing values
    :param strat:str='mean': Specify the strategy to use for imputing missing values
    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_=train.copy(),test.copy()
    
    train,test=train.loc[:, train.columns != target],test.loc[:, test.columns != target]
    
    Input_Cols= list(train.columns)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy=strat)
    imp_mean.fit(train)
    
    train = imp_mean.transform(train[Input_Cols])
    train = pd.DataFrame(train, columns = Input_Cols)
    
    test=imp_mean.transform(test[Input_Cols])
    test = pd.DataFrame(test, columns = Input_Cols)
    
    train[target]=train_[target]
    test[target]=test_[target]
    
    return train,test

def knn_null_imputation(train:pd.DataFrame, 
                        test:pd.DataFrame, 
                        target:str, 
                        neighbors:int=5):
    '''
    The knn_null_imputation function imputes the null values in a dataframe using KNN imputation. 
    The function returns two dataframes, one with the null values imputed and another with the target column unchanged.
    
    
    :param train:pd.DataFrame: Specify the training dataset
    :param test:pd.DataFrame: Specify the dataframe that will be used to impute the missing values
    :param target:str: Specify the target variable
    :param neighbors:int=5: Set the number of neighbors to be used in the knn algorithm
    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_=train.copy(),test.copy()

    train,test=train.loc[:, train.columns != target],test.loc[:, test.columns != target]
    imputer = KNNImputer(n_neighbors=neighbors)
    imputer.fit(train)
    train = pd.DataFrame(imputer.transform(train),columns = train.columns) 
    test = pd.DataFrame(imputer.transform(test),columns = test.columns) 
    
    train[target],test[target]=train_[target],test_[target]
    
    return train,test

def iterative_null_imputation(train:pd.DataFrame, 
                              test:pd.DataFrame, 
                              target:str, 
                              order:str='ascending', 
                              iterations:int=10):
    '''
    
    The iterative_null_imputation function imputes null values in the dataframe. 
    The imputation is done by iteratively replacing null values with the mean of that column. 
    This function takes three parameters: train, test, and target. The train parameter is a pandas dataframe containing all training data (including target variable). The test parameter is a pandas dataframe containing all test/validation/holdout set (including target variable). The target parameter specifies which column contains the target variable for this dataset.
    
    :param train:pd.DataFrame: Specify the training dataframe
    :param test:pd.DataFrame: Impute the missing values in test
    :param target:str: Specify the target variable
    :param order:str='ascending': Determine the order in which the imputation is done
    :param iterations:int=10: Specify the number of iterations that the iterative imputation algorithm will run

    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_=train.copy(),test.copy()
    
    train,test=train.loc[:, train.columns != target],test.loc[:, test.columns != target]
    
    Input_Cols= list(train.columns)
    
    imputer = IterativeImputer(imputation_order=order,max_iter=iterations,random_state=0,n_nearest_features=None)#(int(len(Input_Cols)*0.2)))
    imputer=imputer.fit(train)
    
    train = pd.DataFrame(imputer.transform(train))
    train.columns = Input_Cols
    
    test = pd.DataFrame(imputer.transform(test))
    test.columns = Input_Cols

    train[target],test[target]=train_[target],test_[target]
    
    return train,test

def null_substitution_method(train:pd.DataFrame, 
                             test:pd.DataFrame, 
                             target:str, 
                             enc_method:str, 
                             pred_type:str,
                             eval_metric:str):
    
    '''
    The null_substitution_method function takes in a train and test dataframe, 
    the target variable, the encoding method used to encode categorical variables, 
    the predictive type (regression or classification), and the evaluation metric.  
    The null_substitution_method function returns a list of MAEs for each imputation algorithm.  
    It also returns an Imputation Method that was chosen based on which one had the lowest MAEs.

    :param train:pd.DataFrame: Pass the training data
    :param test:pd.DataFrame: test the function with a small dataset
    :param target:str: Specify the target column
    :param enc_method:str: Choose the encoding method used to encode categorical variables
    :param pred_type:str: Specify if the problem is a regression or classification problem
    :param eval_metric:str: Select the evaluation metric used to rank the different imputation methods
    '''

    train_,test_=train.copy(),test.copy()
    sel_cols= list(train.columns)

    num_cols,cat_cols=numerical_columns(train,target),categorical_columns (train,target) 

    if pred_type=='Reg':
        metric='MAE'
    elif pred_type=='Class':
        metric='AUC'
    
    if enc_method=='Encoding Version 1':
        train_pred,test_pred,train, test=version1_encoding(train, test,target)
    elif enc_method=='Encoding Version 2':
        train_pred,test_pred,train, test=version2_encoding(train, test,target)
    elif enc_method=='Encoding Version 3':
        train_pred,test_pred,train, test=version3_encoding(train, test,target)
    elif enc_method=='Encoding Version 4':
        train_pred,test_pred,train, test=version4_encoding(train, test,target)
        
    print('Const Null Substitution Loading')
    train_const,test_const=const_null_imputation(train,test,target)
    
    print('Simple Imputation Loading')
    train_simple, test_simple=simple_null_imputation(train,test,target)

    print('KNN Imputation Loading')
    train_knn, test_knn = knn_null_imputation(train,test,target)
    
    print('Iterative Imputation Loading')
    train_iter,test_iter=iterative_null_imputation(train,test,target)

    const_perf=pred_eval(train_const, test_const,target)
    simple_perf=pred_eval(train_simple, test_simple,target)
    knn_perf=pred_eval(train_knn, test_knn,target) 
    iter_perf=pred_eval(train_iter, test_iter,target)

    List=[knn_perf,iter_perf,const_perf,simple_perf]
    perf_imp=pd.concat(List)
    perf_imp=perf_imp.reset_index()
    perf_imp = perf_imp.sort_values([eval_metric], ascending=True)
    
    mae_const=const_perf[eval_metric].sum()
    mae_simple=simple_perf[eval_metric].sum()
    mae_knn=knn_perf[eval_metric].sum()
    mae_Iterartive=iter_perf[eval_metric].sum()
    
    print('Null Imputation Methods Performance:')
    
    print('KNN Performance: ', round(mae_knn, 5), '\n Iterative Performance: ', round(mae_Iterartive, 5),
          '\n Const Performance: ', round(mae_const, 5), '\n Simple Performance: ', round(mae_simple, 5))
    
    list_imp=[mae_Iterartive,mae_knn,mae_const,mae_simple]
    
    list_imp.sort()
    imp_method=''
    
    if pred_type=='Class':
        list_imp.sort(reverse=True)

    if list_imp[0]==mae_Iterartive:
        imp_method='Iterative'
        train, test=train_iter.copy(),test_iter.copy()
        print('Iterative Imputation Algorithm was chosen with an ', metric, ' of: ', round(mae_Iterartive, 5))
    elif list_imp[0]==mae_knn:
        imp_method='KNN'
        train, test=train_knn.copy(), test_knn.copy()
        print('KNN Imputation Algorithm was chosen with an ', metric, ' of: ', round(mae_knn, 5))
    elif list_imp[0]==mae_const:
        imp_method='Const'   
        train, test=train_const.copy(),test_const.copy()
        print('Constant Imputation was chosen with an ', metric, ' of: ', round(mae_const, 5))
    elif list_imp[0]==mae_simple:
        imp_method='Simple'  
        train, test=train_simple.copy(),test_simple.copy()
        print('Simple  Imputation Algorithm was chosen with an ', metric, ' of: ', round(mae_simple, 5))

    return train, test,list_imp,imp_method,perf_imp

############################################################ ENCODINGS #########################################################################

def encode_idf(train: pd.DataFrame, test: pd.DataFrame,target:str) -> tuple: ### target= Nome Coluna target
    
    '''
    The encode_idf function takes a dataframe and encodes the categorical columns using the IDF method.
    The function returns two transformed datasets, one for training and another for testing.
    
    :param train:pd.DataFrame: Select the training dataset
    :param test:pd.DataFrame: Apply the transformation to a dataset that is not part of the training set
    :param target:str: Specify the target column
    :return: A tuple containing the transformed train and test datasets
    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_ = train.copy(),test.copy()
    _train,_test = train_.copy(),test_.copy()
    
    encoders=categorical_columns(_train,target)
    
    if len(encoders)>0:
        
        IDF_filter = cane.idf(_train, n_coresJob=2,disableLoadBar = False, columns_use = encoders)  # application of specific multicolumn setting IDF
        idfDicionary = cane.idfDictionary(Original = _train, Transformed = IDF_filter, columns_use = encoders) #, targetColumn=target)
        
        for col in encoders:
            _test[col] = (_test[col]
                             .map(idfDicionary[col])                  
                             .fillna(max(idfDicionary[col].values()))) #self.idf_dict -> dici IDF
                            
        for col in encoders:
            _train[col]=IDF_filter[col]

        _train=transform_dataset(train_,_train[encoders])
        _test=transform_dataset(test_,_test[encoders])
    else:
        print('###### No Categorical Columns ######')

    return _train,_test

def encode_label(train: pd.DataFrame, test: pd.DataFrame, target:str) -> tuple:
    
    '''
    The encode_label function takes a DataFrame and encodes the categorical columns.
    It returns two DataFrames, one with the encoded data and another with the original data.
    
    :param train:pd.DataFrame: Pass the training dataset
    :param test:pd.DataFrame: Encode the test dataset
    :param target:str: Specify the target column
    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_ = train.copy(),test.copy()
    _train,_test = train_.copy(),test_.copy()
    
    _encoders = []
    encoders=categorical_columns(_train,target)
    
    if len(encoders)>0:

        for enc in encoders:
            values = _train[enc].unique()
            _values = pd.Series(values).dropna().reset_index(drop=True)
            _values.index += 1
            _values.name = enc
            _encoders.append(_values)
            dict_values = dict(map(reversed, _values.to_dict().items()))
    
            _train[enc] = _train[enc].map(dict_values.get, na_action='ignore')
            values_test = _test[enc]
            values_test = pd.Series(
                values_test[~values_test.isin(_values)].unique()
            ).dropna()
            _values = pd.concat([_values, values_test], ignore_index=True)
            _values.index += 1
            _values.name = enc
            dict_values = dict(map(reversed, _values.to_dict().items()))
            _test[enc] = _test[enc].map(dict_values.get)
        encoding_dict = pd.concat(_encoders, axis=1)

        _train=transform_dataset(train_,_train[encoders])
        _test=transform_dataset(test_,_test[encoders])
    else:
        print('###### No Categorical Columns ######')

    return _train, _test

def encode_standard(train: pd.DataFrame, test: pd.DataFrame, target:str) -> tuple:
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_=train.copy(),test.copy()
    
    encoders=numerical_columns(train,target)
    
    if len(encoders)>0:
        
        scaler,num_cols=fit_StandardScaler(train_,target)
        train_[num_cols] = scaler.transform(train_[num_cols])
        test_[num_cols] = scaler.transform(test_[num_cols])

    return train_,test_

def encode_minmax(train: pd.DataFrame, test: pd.DataFrame, target:str) -> tuple:
       
    train,test=reset_index_DF(train),reset_index_DF(test)
    train_,test_=train.copy(),test.copy()
    
    encoders=numerical_columns(train,target)
    
    if len(encoders)>0:
        
        scaler,num_cols = fit_MinmaxScaler(train_,target)
        train_[num_cols] = scaler.transform(train_[num_cols])
        test_[num_cols] = scaler.transform(test_[num_cols])

    return train_,test_

#############################################################  Encodings Validation   ###############################################################################

def version1_encoding(train:pd.DataFrame, test:pd.DataFrame, target:str):
    
    '''
    The version_encoding functions take in a training and test dataframe, as well as the name of the target column. 
    It then encodes all categorical columns using one-hot encoding, and all numerical columns using standard scaling. 
    If there are any null values in either the training or test set, it will impute them with 0s for numerical values and 'MISSING' for categorical ones.

    :param train:pd.DataFrame: Specify the training dataset
    :param test:pd.DataFrame: Check if there are any missing values in the test data
    :param target:str: Specify the target column name
    '''

    train_,test_ = train.copy(),test.copy()
    _train,_test = train_.copy(),test_.copy()

    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 

    if len(num_cols)>0:
        _train,_test=encode_standard(_train,_test,target)
    if len(cat_cols)>0:
        _train,_test=encode_idf(_train,_test,target)

    if (_train.isnull().sum().sum() or _test.isnull().sum().sum()) != 0:
        train_pred, test_pred = simple_null_imputation(_train,_test,target)
    else:
        train_pred, test_pred=_train.copy(), _test.copy()
    
    return train_pred, test_pred,_train,_test

def version2_encoding(train:pd.DataFrame, test:pd.DataFrame, target:str):
        
    train_,test_ = train.copy(),test.copy()
    _train,_test = train_.copy(),test_.copy()

    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 

    if len(num_cols)>0:
        _train,_test=encode_minmax(_train,_test,target)    
    if len(cat_cols)>0:
        _train,_test=encode_idf(_train,_test,target)

    if (_train.isnull().sum().sum() or _test.isnull().sum().sum()) != 0:
        train_pred, test_pred = simple_null_imputation(_train,_test,target)
    else:
        train_pred, test_pred=_train.copy(), _test.copy()
    
    return train_pred, test_pred,_train,_test

def version3_encoding(train:pd.DataFrame, test:pd.DataFrame, target:str):
    
    train_,test_ = train.copy(),test.copy()
    _train,_test = train_.copy(),test_.copy()

    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 
    
    if len(num_cols)>0:
        _train,_test=encode_standard(_train,_test,target)
    if len(cat_cols)>0:
        _train,_test=encode_label(_train,_test,target)
                                           
    if (_train.isnull().sum().sum() or _test.isnull().sum().sum()) != 0:
        train_pred, test_pred = simple_null_imputation(_train,_test,target)
    else:
        train_pred, test_pred=_train.copy(), _test.copy()
    
    return train_pred, test_pred,_train,_test

def version4_encoding(train:pd.DataFrame, test:pd.DataFrame, target:str):
       
    train_,test_ = train.copy(),test.copy()
    _train,_test = train_.copy(),test_.copy()

    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 
    
    if len(num_cols)>0:
        _train,_test=encode_minmax(_train,_test,target)
    if len(cat_cols)>0:
        _train,_test=encode_label(_train,_test,target)
    if (_train.isnull().sum().sum() or _test.isnull().sum().sum()) != 0:
        train_pred, test_pred = simple_null_imputation(_train,_test,target)
    else:
        train_pred, test_pred=_train.copy(), _test.copy()
    
    return train_pred, test_pred,_train,_test

##########################################################  Select Best Encoding Method  ############################################################################

def Select_Encoding_Method(train:pd.DataFrame, 
                           test:pd.DataFrame, 
                           target:str, 
                           pred_type:str,
                           eval_metric:str):
    '''
    The Select_Encoding_Method function is used to select the best encoding method for a given dataset. 
    The function takes in 5 parameters:
        1) train: The training dataframe that will be used to train the model. This is a pandas dataframe object.
        2) test: The test dataframe that will be used to test the performance of our model on unseen data. This is a pandas 
                    dataframe object.    
        3) target: A string indicating which column name represents the target variable in your dataset (y).  
        4) pred_type: A string indicating whether
    
    :param train:pd.DataFrame: Select the train dataframe
    :param test:pd.DataFrame: test the model with a different dataset than the one used for training
    :param target: Select the target variable
    :param pred_type:str: Select between regression or classification
    :param eval_metric:str: Select the evaluation metric used to compare the different encoding versions
    :return: The train and test datasets with the best encoding method
    '''
    
    train_,test_=train.copy(),test.copy()
    _train_,_test_=train_.copy(),test_.copy()

    
    train_v1_p,test_v1_p,train_v1, test_v1=version1_encoding(_train_, _test_,target)
    train_v2_p,test_v2_p,train_v2, test_v2=version2_encoding(_train_, _test_,target)
    train_v3_p,test_v3_p,train_v3, test_v3=version3_encoding(_train_, _test_,target)
    train_v4_p,test_v4_p,train_v4, test_v4=version4_encoding(_train_, _test_,target)

    pred_perf_v1=pred_eval(train_v1_p, test_v1_p,target)
    pred_perf_v2=pred_eval(train_v2_p, test_v2_p,target)
    pred_perf_v3=pred_eval(train_v3_p, test_v3_p,target)
    pred_perf_v4=pred_eval(train_v4_p, test_v4_p,target)
    
    p_v1=pred_perf_v1[eval_metric].sum()
    p_v2=pred_perf_v2[eval_metric].sum()
    p_v3=pred_perf_v3[eval_metric].sum()
    p_v4=pred_perf_v4[eval_metric].sum()
    
    if pred_type=='Reg':
        print(' ')
        print('Predictive Performance Encoding Versions:')
        print(' ')
        print('\n MAE Version 1 [IDF + StandardScaler] : ', round(p_v1, 5),
              '\n MAE Version 2 [IDF + MinMaxScaler] : ', round(p_v2, 5),
              '\n MAE Version 3 [Label + StandardScaler] : ', round(p_v3, 5),
              '\n MAE Version 4 [IDF + MinMaxScaler] : ', round(p_v4, 5))
        metric='MAE'
    elif pred_type=='Class':
        print('Predictive Performance Encoding Versions:')
        print('\n AUC Version 1 [IDF + StandardScaler] : ', round(p_v1, 5),
              '\n AUC Version 2 [IDF + MinMaxScaler] : ', round(p_v2, 5),
              '\n AUC Version 3 [Label + StandardScaler] : ', round(p_v3, 5),
              '\n AUC Version 4 [IDF + MinMaxScaler] : ', round(p_v4, 5))
        metric='AUC'
    
    list_encoding=[p_v1,p_v2,p_v3,p_v4]
    list_encoding.sort()
    enc_method=''
    
    if pred_type=='Class':
        list_encoding.sort(reverse=True)
        
    if list_encoding[0]==p_v1:
        enc_method='Encoding Version 1'
        print('Encoding Version 1 was choosen with an ', metric, ' of: ', round(p_v1, 5))
        _train_,_test_=train_v1, test_v1
        
    elif list_encoding[0]==p_v2:
        enc_method='Encoding Version 2'
        print('Encoding Version 2 was choosen with an ', metric, ' of: ', round(p_v2, 5))
        _train_,_test_=train_v2, test_v2        
    
    elif list_encoding[0]==p_v3:
        enc_method='Encoding Version 3'
        print('Encoding Version 3 was choosen with an ', metric, ' of: ', round(p_v3, 5))
        _train_,_test_=train_v3, test_v3
    
    elif list_encoding[0]==p_v4:
        enc_method='Encoding Version 4'
        print('Encoding Version 4 was choosen with an ', metric, ' of: ', round(p_v4, 5))
        _train_,_test_=train_v4, test_v4
    
    return _train_,_test_,enc_method

################################################################## Encodings ###############################################################

############################################################# Encoding Pipeline Methods ####################################################

def encoding_v1(train:pd.DataFrame, test:pd.DataFrame, target:str):
    
    '''
    The encoding functions take in a training and test dataframe, as well as the name of the target column.
    It returns two modified dataframes with all categorical columns encoded using frequency-inverse document 
    frequency encoding. The function will only encode those columns that are categorical (as determined by the 
    categorical_columns function). If there are numerical columns, they will be encoded using standard scaling.
    
    :param train:pd.DataFrame: Specify the training dataframe
    :param test:pd.DataFrame: Encode the test dataframe
    :param target:str: Specify the target column
    '''
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    _train,_test = train.copy(),test.copy()
    
    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 

    if len(num_cols)>0:
        _train,_test=encode_standard(_train,_test,target)    
    if len(cat_cols)>0:
        _train,_test=encode_idf(_train,_test,target)

    return _train,_test

def encoding_v2(train:pd.DataFrame, test:pd.DataFrame, target:str):
        
    train,test=reset_index_DF(train),reset_index_DF(test)
    _train,_test = train.copy(),test.copy()
    
    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 

    if len(num_cols)>0:
        _train,_test=encode_minmax(_train,_test,num_cols)
    if len(cat_cols)>0:
        _train,_test=encode_idf(_train,_test,target)

    return _train,_test

def encoding_v3(train:pd.DataFrame, test:pd.DataFrame, target:str):
    
    train,test=reset_index_DF(train),reset_index_DF(test)
    _train,_test = train.copy(),test.copy()
    
    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target) 
    
    if len(num_cols)>0:
        _train,_test=encode_standard(_train,_test,target)
    if len(cat_cols)>0:
        _train,_test=encode_label(_train,_test,target)
    
    return _train,_test

def encoding_v4(train:pd.DataFrame, test:pd.DataFrame, target:str):
        
    train,test=reset_index_DF(train),reset_index_DF(test)
    _train,_test = train.copy(),test.copy()
    
    num_cols,cat_cols=numerical_columns(_train,target),categorical_columns(_train,target)

    if len(num_cols)>0:
        _train,_test=encode_minmax(_train,_test,target)
    if len(cat_cols)>0:
        _train,_test=encode_label(_train,_test,target)
    
    return _train,_test

######################################################## Encoding Updated Methods - >0.0.9 Version ###########################################

######### MultiColumn LabelEncoding

def fit_Label_Encoding(Dataset:pd.DataFrame,target:str):
    
    encoders=categorical_columns(Dataset,target)
    df,list_cols,list_le=Dataset.copy(),[],[]
    
    for c in encoders:
        le = LabelEncoder()
        list_cols.append(c),list_le.append(le.fit(df[c]))
    le_dict = {list_cols[i]: list_le[i] for i in range(len(list_cols))}
    
    return le_dict

def transform_Label_Encoding(Dataset:pd.DataFrame,le_fit:dict):
    
    encoders=list(le_fit.keys())
    df=Dataset.copy()

    for c in encoders:
        le=le_fit[c]  
        df[c] = df[c].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        df[c] = le.transform(df[c])
        
    return df 

######### MultiColumn OneHotEncoding

def fit_OneHot_Encoding(Dataset:pd.DataFrame,target:str,n_distinct:int=10):

    df,list_cols,list_le=Dataset.copy(),[],[]
    drop_org_cols,list_ohe=True,[] 
    encoders=categorical_columns(df,target)
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
                print("****************", enc)
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
    
    df=Dataset.copy()
    
    num_cols=numerical_columns(df,target)
    
    scaler = StandardScaler()

    scaler = scaler.fit(df[num_cols])

    return scaler, num_cols

def fit_MinmaxScaler(Dataset:pd.DataFrame,target:str):
    
    df=Dataset.copy()
    
    num_cols=numerical_columns(df,target)
    
    scaler = MinMaxScaler() 

    scaler = scaler.fit(df[num_cols])

    return scaler, num_cols

def fit_RobustScaler(Dataset:pd.DataFrame,target:str):
    
    df=Dataset.copy()
    
    num_cols=numerical_columns(df,target)
    
    scaler = RobustScaler()
    
    scaler = scaler.fit(df[num_cols])
    
    return scaler, num_cols

########################################################### Feature Selection ##############################################################

###################################  H2O Feature Selection ######################################

def feature_selection_h2o(Dataset:pd.DataFrame, target:str, total_vi :float=0.98, h2o_fs_models:int =7, encoding_fs:bool=True):
    '''
    The feature_selection_h2o function is used to select the most important input variables for a given model.
    The function takes as input: 
        - Dataset: A pandas dataframe with all the columns of your dataset, including target variable and features. 
        - target: The name of your target variable (string). 
        - Total_vi : The total relative importance percentage you want to keep in your dataset (float). It should be between 0.5 and 1.0 . Default value is 0.98 .  
    
    :param Dataset:pd.DataFrame: Input the dataset
    :param target:str: Define the target column of the dataset
    :param total_vi:float=0.98: Define the minimum percentage of relative importance that will be used to select the input columns
    :param h2o_fs_models:int=7: Define the number of models to be used in the h2o automl process
    :param encoding_fs:bool=True: Encode categorical variables
    '''
    
    assert total_vi>=0.5 and total_vi<=1 , 'total_vi value should be in [0.5,1] interval'
    assert h2o_fs_models>=1 and h2o_fs_models<=50 , 'h2o_fs_models value should be in [0,50] interval'
    
    train_=Dataset.copy()
    train=train_.copy()
    
    if encoding_fs==True:
        le =LabelEncoder()
        cols=categorical_columns(train_,target)   
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
    
    :param X: Specify the dataframe that contains all the independent variables
    :return: A dataframe with the variables and their respective vifs
    '''
    # Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.sort_values(['VIF'], ascending=False)
    return vif

def feature_selection_vif(Dataset:pd.DataFrame, target:str, VIF:float=10.0):
    
    '''
    The feature_selection_vif function takes a pandas dataframe and the name of the target column as input.
    It then calculates VIF for all columns in the dataset and returns a list of selected features based on 
    VIF value. The function also returns vif_df which is used to plot graph between Variance Inflation Factor 
    and independent variables.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param target:str: Specify the target variable
    :param VIF:float=10.0: Set the threshold for vif value
    '''
    
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
    
    '''
    The vif_performance_selection function is used to select the best features from a dataset. 
    It uses the VIF (Variance Inflation Factor) method to remove columns with high multicollinearity. 
    The function takes in 5 parameters: train, test, target, vif_ratio and pred_type. 
    The train parameter is a pandas dataframe of training data containing all the features and target variable(s). 
    The test parameter is a pandas dataframe of testing/validation data containing all the features and target variable(s). 

    :param vif_ratio:float=10.0: Set the threshold for vif (variance inflation factor) value
    :return: The train and test datasets after feature selection
    '''
    
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
    
    '''
    The metrics_regression function calculates the metrics of a regression model.
    It takes as input two arrays: y_real and y_pred, which are the real values and 
    the predicted values of a target variable respectively. It returns a dictionary 
    with all the metrics.
    
    :param y_real: Store the real values of the target variable
    :param y_pred: Predict the values of y based on the model
    :return: A dictionary with the metrics of regression
    '''
    
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
    
    '''
    The metrics_classification function takes in two parameters: y_true and y_pred. 
    It returns a dictionary of metrics for the model's performance on the test set.
    
    :param y_true: Pass the actual labels of the data
    :param y_pred: Store the predicted values
    :return: A dictionary containing the accuracy, precision and f-score
    '''

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
    
    '''
    The metrics_binary_classification function takes in two parameters: y_true and y_pred. 
    It returns a dictionary of metrics for the binary classification task.

    :param y_true: Specify the true class labels of the input samples
    :param y_pred: Pass the predicted values of the target variable
    :return: A dictionary of the metrics that are used to evaluate binary classification problems
    '''

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
    
    list_estimators,rf,et=[100,250,500],[],[]
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

def atlantic_data_processing(Dataset:pd.DataFrame,
                             target:str, 
                             Split_Racio:float,
                             total_vi:float=0.98,
                             h2o_fs_models:int =7,
                             encoding_fs:bool=True,
                             vif_ratio:float=10.0):
    
    '''
    The atlantic_data_processing function is used to preprocess the input dataframe. 
    It is composed of several steps:
        1) Remove columns with more than 99% of null values;
        2) Remove target Null Values;
        3) Datetime Feature Engineering
        4) Feature Selection by Variance H2O AutoML Variable Importance; 
        5) Encoding Method Selection;
        6) NULL SUBSTITUTION + ENCODING APLICATION;
        7) Feature Selection by Variance Inflation Factor (VIF); 
    
       The function returns a Dataframe, train and test DataFrames transformed with the best performance selected preprocessing methods.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be processed
    :param target:str: Define the target column
    :param Split_Racio:float: Define the size of the test set
    :param total_vi:float=0.98: Select the most relevant features
    :param h2o_fs_models:int=7: Select the number of models used in the feature selection process
    :param encoding_fs:bool=True: Select the encoding method
    :param vif_ratio:float=10.0: Control the vif ratio
    '''
    
    Dataframe_=Dataset.copy()
    Dataset_=Dataframe_.copy()

############################## Validation Dataframe ##############################

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

    tr_bk,te_bk,enc_method=Select_Encoding_Method(train,test,target,Pred_type,Eval_metric)

############################## NULL SUBSTITUTION + ENCODING APLICATION ##############################

    if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
        train, test,list_imp,imp_method,perf_imp=null_substitution_method(train,test,target,enc_method,Pred_type,Eval_metric)
    else:
        ## Encoding Method
        imp_method='Undefined'
        if enc_method=='Encoding Version 1':
            train_pred,test_pred,train, test=version1_encoding(train, test,target)
        elif enc_method=='Encoding Version 2':
            train_pred,test_pred,train, test=version2_encoding(train, test,target)
        elif enc_method=='Encoding Version 3':
            train_pred,test_pred,train, test=version3_encoding(train, test,target)
        elif enc_method=='Encoding Version 4':
            train_pred,test_pred,train, test=version4_encoding(train, test,target)
            
        print('    ')   
        print('There are no missing values in the Input Data')    
        print('    ') 
        
############################## VARIANCE INFLATION FACTOR (VIF) APPLICATION ##############################
    
    train, test=vif_performance_selection(train,test,target)

############################## Transformation Procediment ##############################

    Dataframe=Dataframe_.copy()
    Dataframe=del_nulls_target(Dataframe,target) ## Delete target Null Values 
    Dataframe=engin_date(Dataframe)
    Dataframe=Dataframe[list(train.columns)]
    train_df,test_df=split_dataset(Dataframe,Split_Racio)
    train_df=Dataframe.copy()
    
    if enc_method=='Encoding Version 1':
        train_df, test_df=encoding_v1(train_df, test_df,target)
    elif enc_method=='Encoding Version 2':
        train_df, test_df=encoding_v2(train_df, test_df,target)
    elif enc_method=='Encoding Version 3':
        train_df, test_df=encoding_v3(train_df, test_df,target)
    elif enc_method=='Encoding Version 4':
        train_df, test_df=encoding_v4(train_df, test_df,target)

    if imp_method=='Const':
        train_df,test_df=const_null_imputation(train_df, test_df,target)
    elif imp_method=='Simple':  
        train_df, test_df=simple_null_imputation(train_df, test_df,target)
    elif imp_method=='KNN':  
        train_df, test_df = knn_null_imputation(train_df, test_df,target)
    elif imp_method=='Iterative':
        train_df,test_df=iterative_null_imputation(train_df, test_df,target)

    Dataframe=reset_index_DF(train_df)

    return Dataframe, train, test
