import numpy as np
import pandas as pd
import cane
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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
    
    assert Split_Racio>=0.5 and Split_Racio<=0.95 , "Split_Racio value should be in [0.5,0.95[ interval"
    
    Train, Test= train_test_split(Dataset, train_size=Split_Racio)

    return Train,Test

def transform_dataset(Dataset:pd.DataFrame, Dataset_Transf:pd.DataFrame):
    
    """
    The transform_dataset function takes two datasets, one to be transformed and another that contains the 
    transformations. The transform_dataset function then transforms the dataset with all of its transformations.


    :param Dataset:pd.DataFrame: Pass the dataset to be transformed
    :param Dataset_Transf:pd.DataFrame: Transform the dataset:pd
    :return: A dataset with the transformed columns
    """

    _Dataset = Dataset.copy()
    _Dataset_Transf = Dataset_Transf.copy()

    for Colunas in Dataset_Transf:
        _Dataset[[Colunas]]=_Dataset_Transf[[Colunas]]
        
    return _Dataset


def reindex_columns(Dataset:pd.DataFrame, Feature_Importance:list):
    
    """
    The reindex_columns function takes a dataframe and a list of column names as input. 
    It returns the same dataframe with the columns in the order specified by the list of column names.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be reindexed
    :param Feature_Importance:list: Reindex the columns in the dataframe
    :return: A dataframe with the columns reordered to match the feature importance list
    """
    
    total_cols=list(Dataset.columns)
    y=feature_importance+total_cols
    z=list(dict.fromkeys(y))
    Dataset=Dataset[z]
    
    return Dataset

def target_type(Dataset:pd.DataFrame, Target:str):  
    
    """
    The target_type function takes in a pandas dataframe and returns the type of prediction problem. 
    
    :param Dataset:pd.DataFrame: Pass the dataframe that we want to use for our analysis
    :param Target:str: Identify the target variable
    """

    df=Dataset[[Target]]
    reg_target,class_target=df.select_dtypes(include=['int','float']).columns.tolist(),df.select_dtypes(include=['object']).columns.tolist()
    if len(class_target)==1:
        pred_type="Class"
        eval_metric="Accuracy"
    elif len(reg_target)==1:
        pred_type="Reg"
        eval_metric="Mean Absolute Error"
    return pred_type, eval_metric

############################################################# Datetime Feature Engineering ######################################################

def engin_date(Dataset:pd.DataFrame, Drop:bool=True):
    
    """
    The engin_date function takes a DataFrame as input and returns the same DataFrame with additional date-time features.
    The function will return a new column for each date-time feature generated, along with an updated list of columns.
    
    
    :param Dataset:pd.DataFrame: Pass the dataset to be transformed
    :param Drop:bool=True: Drop the original date columns
    :return: A dataframe with the date features created
    """
    
    Dataset_=Dataset.copy()
    Df=Dataset_.copy()
    
    x=pd.DataFrame(Df.dtypes)
    x['column'] = x.index
    x=x.reset_index().drop(["index"], axis=1).rename(columns={0: 'dtype'})
    a=x.loc[x['dtype'] == 'datetime64[ns]']

    list_date_columns=[]
    for col in a['column']:
        list_date_columns.append(col)

    def create_date_features(df,elemento):
        
        df[elemento + '_day_of_week'] = df[elemento].dt.dayofweek + 1
        df[[elemento + "_is_wknd"]] = df[[elemento + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7], 
                            [0, 0, 0, 0, 0, 1, 1 ]) 
        df[elemento + '_day_of_month'] = df[elemento].dt.day
        df[elemento + '_day_of_year'] = df[elemento].dt.dayofyear
        df[elemento + '_month'] = df[elemento].dt.month
        df[elemento + '_year'] = df[elemento].dt.year
        df[elemento + "_Season"]=""
        # "day of year" ranges for the northern hemisphere    
        winter = list(range(1,80)) + list(range(355,370))
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        df.loc[(df[elemento + "_day_of_year"].isin(spring)), elemento + "_Season"] = "2"
        df.loc[(df[elemento + "_day_of_year"].isin(summer)), elemento + "_Season"] = "3"
        df.loc[(df[elemento + "_day_of_year"].isin(fall)), elemento + "_Season"] = "4"
        df.loc[(df[elemento + "_day_of_year"].isin(winter)), elemento + "_Season"] = "1"
        df[elemento + "_Season"]=df[elemento + "_Season"].astype(np.int64)
        
        return df 
    
    if Drop==True:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
            Df=Df.drop(elemento,axis=1)
    elif Drop==False:
        for elemento in list_date_columns:
            Df=create_date_features(Df,elemento)
    if len(list_date_columns)>=1:
        print("#######  Date Time Feature Generation Implemented  ########")
        
    return Df

############################################################# Enconding Lists ###################################################################

def numerical_columns(Dataset:pd.DataFrame, Target:str):
    
    """
    The numerical_columns function returns a list of the numerical columns in the dataframe.
    It takes two arguments: Dataset and Target. 
    Dataset is the name of your dataset, and Target is the name of your target variable
    
    :param Dataset:pd.DataFrame: Pass the dataframe that will be used to analyze
    :param Target:str: Identify the target variable
    :return: A list with the numerical columns of a dataframe
    """
    
    list_num_cols=Dataset.select_dtypes(include=['int','float']).columns.tolist()
    
    for elemento in list_num_cols:
        if elemento==Target:
            list_num_cols.remove(Target)
            
    return list_num_cols

def categorical_columns(Dataset:pd.DataFrame, Target):
    
    """
    The categorical_columns function returns a list of the categorical columns in the dataset.
    The function takes two arguments: Dataset and Target. 
    Dataset is a pandas dataframe, and Target is the name of your target variable column as a string.
    
    :param Dataset:pd.DataFrame: Specify the dataframe where we want to find the categorical columns
    :param Target: Remove the target column from the list of categorical columns
    :return: A list with the categorical columns of a dataframe
    """
    
    list_cat_cols=Dataset.select_dtypes(include=['object']).columns.tolist()

    for elemento in list_cat_cols:
        if elemento==Target:
            list_cat_cols.remove(Target)
            
    return list_cat_cols 

############################################################# Nulls Treatment ###################################################################

def del_nulls_target(Dataset:pd.DataFrame, Target:str):
        
    Dataset=Dataset[Dataset[Target].isnull()==False]
    
    return Dataset

def remove_columns_by_nulls(Dataset:pd.DataFrame, percentage:int): ## Colunas 
    
    """
    The remove_columns_by_nulls function removes columns from a dataframe that have more than the percentage of null values specified by the user.
    The function takes two arguments:
    Dataset - The dataset to be modified. This should be a pandas DataFrame object.
    percentage - A number between 0 and 100, inclusive, which represents the maximum allowable percentage of null values in any given column before it is removed from the dataframe.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be cleaned
    :param percentage:int: Specify the percentage of null values that a column must have in order to be removed
    """
    
    assert percentage>0 and percentage<=100 , "percentage should not exceed value of 100"
    
    perc = percentage
    min_count =  int(((100-perc)/100)*Dataset.shape[0] + 1)
    Dataset = Dataset.dropna( axis=1,
                thresh=min_count)
    Dataset = Dataset.loc[:, (Dataset==0).mean()*100 < perc]
    Dataset = Dataset.loc[:, ~(Dataset.apply(pd.Series.value_counts, normalize=True).max() > perc/100)]

    return Dataset

############################################################# Null_Substitution ########################################################################################

def const_null_imputation(Train_DF:pd.DataFrame,
                          Test_DF:pd.DataFrame, 
                          Target:str, 
                          imp_value:int=0):
    
    """
    The const_null_imputation function imputes a constant value to all the null values in the dataframe.
    The function takes three arguments: Train_DF, Test_DF and Target. 
    Train_DF is a pandas dataframe that contains training data with null values. 
    Test_DF is a pandas dataframe that contains test/validation/holdout set with null values. 
    Target is the target variable name in both train and test sets.
    
    :param Train_DF:pd.DataFrame: Specify the training dataset
    :param Test_DF:pd.DataFrame: Test the model on a different dataset
    :param Target:str: Specify the target variable
    :param imp_value:int=0: Set the imputation value for all the null values in both train and test dataframes
    :return: The training and test dataframes with the null values imputed by a constant value
    """
    
    Train_DF,Test_DF=reset_index_DF(Train_DF),reset_index_DF(Test_DF)
    train_df_=Train_DF.copy()
    test_df_=Test_DF.copy()
    
    Train_DF,Test_DF=Train_DF.loc[:, Train_DF.columns != Target],Test_DF.loc[:, Test_DF.columns != Target]
    Train_DF,Test_DF=Train_DF.fillna(imp_value),Test_DF.fillna(imp_value)
    Train_DF[Target],Test_DF[Target]=train_df_[Target],test_df_[Target]
    
    return Train_DF,Test_DF

def simple_null_imputation(Train_DF:pd.DataFrame,
                           Test_DF:pd.DataFrame,
                           Target:str,
                           strat:str='mean'):
    
    """
    The simple_null_imputation function takes in a Train_DF and Test_DF, as well as the Target column name. 
    It then removes the target column from both Train_DF and Test_DF, imputes missing values with mean for numerical columns 
    and most frequent value for categorical columns. It returns two dataframes: one with train data and one with test data.
    
    :param Train_DF:pd.DataFrame: Specify the training dataframe
    :param Test_DF:pd.DataFrame: Impute the missing values in the test_df
    :param Target:str: Specify the target variable for which we are imputing missing values
    :param strat:str='mean': Specify the strategy to use for imputing missing values
    """
    
    Train_DF,Test_DF=reset_index_DF(Train_DF),reset_index_DF(Test_DF)
    train_df_=Train_DF.copy()
    test_df_=Test_DF.copy()
    
    Train_DF,Test_DF=Train_DF.loc[:, Train_DF.columns != Target],Test_DF.loc[:, Test_DF.columns != Target]
    
    Cols_Input= list(Train_DF.columns)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy=strat)
    imp_mean.fit(Train_DF)
    
    Train_DF= imp_mean.transform(Train_DF[Cols_Input])
    Train_DF = pd.DataFrame(Train_DF, columns = Cols_Input)
    
    Test_DF=imp_mean.transform(Test_DF[Cols_Input])
    Test_DF = pd.DataFrame(Test_DF, columns = Cols_Input)
    
    Train_DF[Target]=train_df_[Target]
    Test_DF[Target]=test_df_[Target]
    
    return Train_DF,Test_DF

def knn_null_imputation(Train_DF:pd.DataFrame, 
                        Test_DF:pd.DataFrame, 
                        Target:str, 
                        neighbors:int=5):
    """
    The knn_null_imputation function imputes the null values in a dataframe using KNN imputation. 
    The function returns two dataframes, one with the null values imputed and another with the target column unchanged.
    
    
    :param Train_DF:pd.DataFrame: Specify the training dataset
    :param Test_DF:pd.DataFrame: Specify the dataframe that will be used to impute the missing values
    :param Target:str: Specify the target variable
    :param neighbors:int=5: Set the number of neighbors to be used in the knn algorithm
    """
    
    Train_DF,Test_DF=reset_index_DF(Train_DF),reset_index_DF(Test_DF)
    train_df_=Train_DF.copy()
    test_df_=Test_DF.copy()
    Train_DF,Test_DF=Train_DF.loc[:, Train_DF.columns != Target],Test_DF.loc[:, Test_DF.columns != Target]
    imputer = KNNImputer(n_neighbors=neighbors)
    imputer.fit(Train_DF)
    Train_DF = pd.DataFrame(imputer.transform(Train_DF),columns = Train_DF.columns) 
    Test_DF = pd.DataFrame(imputer.transform(Test_DF),columns = Test_DF.columns) 
    
    Train_DF[Target],Test_DF[Target]=train_df_[Target],test_df_[Target]
    
    return Train_DF,Test_DF

def iterative_null_imputation(Train_DF:pd.DataFrame, 
                              Test_DF:pd.DataFrame, 
                              Target:str, 
                              order:str='ascending', 
                              iterations:int=10):
    """
    
    The iterative_null_imputation function imputes null values in the dataframe. 
    The imputation is done by iteratively replacing null values with the mean of that column. 
    This function takes three parameters: Train_DF, Test_DF, and Target. The Train_DF parameter is a pandas dataframe containing all training data (including target variable). The Test_DF parameter is a pandas dataframe containing all test/validation/holdout set (including target variable). The Target parameter specifies which column contains the target variable for this dataset.
    
    :param Train_DF:pd.DataFrame: Specify the training dataframe
    :param Test_DF:pd.DataFrame: Impute the missing values in test_df
    :param Target:str: Specify the target variable
    :param order:str='ascending': Determine the order in which the imputation is done
    :param iterations:int=10: Specify the number of iterations that the iterative imputation algorithm will run

    """
    
    Train_DF,Test_DF=reset_index_DF(Train_DF),reset_index_DF(Test_DF)
    train_df_=Train_DF.copy()
    test_df_=Test_DF.copy()
    Train_DF,Test_DF=Train_DF.loc[:, Train_DF.columns != Target],Test_DF.loc[:, Test_DF.columns != Target]
    
    Cols_Input= list(Train_DF.columns)
    
    imputer = IterativeImputer(imputation_order=order,max_iter=iterations,random_state=0,n_nearest_features=None)#(int(len(Cols_Input)*0.2)))
    imputer=imputer.fit(Train_DF)
    
    Train_DF = pd.DataFrame(imputer.fit_transform(Train_DF))
    Train_DF.columns = Cols_Input
    
    Test_DF = pd.DataFrame(imputer.fit_transform(Test_DF))
    Test_DF.columns = Cols_Input

    Train_DF[Target],Test_DF[Target]=train_df_[Target],test_df_[Target]
    
    return Train_DF,Test_DF

def null_substitution_method(train:pd.DataFrame, 
                             test:pd.DataFrame, 
                             Target:str, 
                             Encoding_Method:str, 
                             pred_type:str,
                             eval_metric:str):
    
    """
    The null_substitution_method function takes in a train and test dataframe, 
    the target variable, the encoding method used to encode categorical variables, 
    the predictive type (regression or classification), and the evaluation metric.  
    The null_substitution_method function returns a list of MAEs for each imputation algorithm.  
    It also returns an Imputation Method that was chosen based on which one had the lowest MAEs.

    :param train:pd.DataFrame: Pass the training data
    :param test:pd.DataFrame: Test the function with a small dataset
    :param Target:str: Specify the target column
    :param Encoding_Method:str: Choose the encoding method used to encode categorical variables
    :param pred_type:str: Specify if the problem is a regression or classification problem
    :param eval_metric:str: Select the evaluation metric used to rank the different imputation methods
    """

    Train_=train.copy()
    Test_=test.copy()
    Train=Train_.copy()
    Test=Test_.copy()
    
    Selected_Cols= list(Train.columns)

    list_num_cols=numerical_columns(Train,Target)  
    list_cat_cols=categorical_columns (Train,Target) 
    Input_Cols=list_num_cols+list_cat_cols
    
    if pred_type=="Reg":
        metric="MAE"
    elif pred_type=="Class":
        metric="AUC"
    
    if Encoding_Method=="Encoding Version 1":
        Train_Pred,Test_Pred,Train, Test=version1_Encoding(Train, Test,Target)
    elif Encoding_Method=="Encoding Version 2":
        Train_Pred,Test_Pred,Train, Test=version2_Encoding(Train, Test,Target)
    elif Encoding_Method=="Encoding Version 3":
        Train_Pred,Test_Pred,Train, Test=version3_Encoding(Train, Test,Target)
    elif Encoding_Method=="Encoding Version 4":
        Train_Pred,Test_Pred,Train, Test=version4_Encoding(Train, Test,Target)
        
    print("Zero Null Substitution Loading")
    Train_Zero_Sub,Test_Zero_Sub=const_null_imputation(Train,Test,Target)
    
    print("Simple Imputation Loading")
    Train_Simple, Test_Simple=simple_null_imputation(Train,Test,Target)

    print("KNN Imputation Loading")
    Train_KNN, Test_KNN = knn_null_imputation(Train,Test,Target)
    
    print("Iterative Imputation Loading")
    Train_Iterative,Test_Iterative=iterative_null_imputation(Train,Test,Target)

    Zero_Sub_Performance=Predictive_Evaluation(Train_Zero_Sub, Test_Zero_Sub,Target,pred_type)
    Simple_Performance=Predictive_Evaluation(Train_Simple, Test_Simple,Target,pred_type)
    KKN_Performance=Predictive_Evaluation(Train_KNN, Test_KNN,Target,pred_type) 
    Iterative_Performance=Predictive_Evaluation(Train_Iterative, Test_Iterative,Target,pred_type)

    List=[KKN_Performance,Iterative_Performance,Zero_Sub_Performance,Simple_Performance]
    Performance_Imputation_Algorithms=pd.concat(List)
    Performance_Imputation_Algorithms=Performance_Imputation_Algorithms.reset_index()
    Performance_Imputation_Algorithms = Performance_Imputation_Algorithms.sort_values([eval_metric], ascending=True)
    
    MAE_Zero_Sub=Zero_Sub_Performance[eval_metric].sum()
    MAE_Simple=Simple_Performance[eval_metric].sum()
    MAE_KNN=KKN_Performance[eval_metric].sum()
    MAE_Iterartive=Iterative_Performance[eval_metric].sum()

    print("KNN Performance: ", KKN_Performance, "\n Iterative Performance: ", Iterative_Performance,
          "\n Zero Substitution Performance: ", Zero_Sub_Performance, "\n Simple Imputation Performance: ", Simple_Performance)
    List_Imputation=[MAE_Iterartive,MAE_KNN,MAE_Zero_Sub,MAE_Simple]
    
    List_Imputation.sort()
    Imputation_Method=""

    if List_Imputation[0]==MAE_Iterartive:
        Imputation_Method="Iterative"
        Train, Test=Train_Iterative,Test_Iterative
        print("Iterative Imputation Algorithm was chosen with an ", metric, " of: ", MAE_Iterartive)
    elif List_Imputation[0]==MAE_KNN:
        Imputation_Method="KNN"
        Train, Test=Train_KNN, Test_KNN
        print("KNN Imputation Algorithm was chosen with an ", metric, " of: ", MAE_KNN)
    elif List_Imputation[0]==MAE_Zero_Sub:
        Imputation_Method="Zero_Sub"   
        Train, Test=Train_Zero_Sub,Test_Zero_Sub
        print("Zero (Constant) Imputation was chosen with an ", metric, " of: ", MAE_Zero_Sub)
    elif List_Imputation[0]==MAE_Simple:
        Imputation_Method="Simple"  
        Train, Test=Train_Simple,Test_Simple
        print("Simple  Imputation Algorithm was chosen with an ", metric, " of: ", MAE_Simple)

    return Train, Test,List_Imputation,Imputation_Method,Performance_Imputation_Algorithms

############################################################ ENCODINGS #########################################################################

def encode_idf(df_train: pd.DataFrame, df_test: pd.DataFrame,target:str) -> tuple: ### Target= Nome Coluna target
    
    """
    The encode_idf function takes a dataframe and encodes the categorical columns using the IDF method.
    The function returns two transformed datasets, one for training and another for testing.
    
    :param df_train:pd.DataFrame: Select the training dataset
    :param df_test:pd.DataFrame: Apply the transformation to a dataset that is not part of the training set
    :param target:str: Specify the target column
    :return: A tuple containing the transformed train and test datasets
    """
    
    df_train=reset_index_DF(df_train)
    df_test=reset_index_DF(df_test)
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()
    _df_train = df_train_.copy()
    _df_test = df_test_.copy()
    
    encoders=categorical_columns(_df_train,target)
    
    if len(encoders)>0:
        
        IDF_filter = cane.idf(_df_train, n_coresJob=2,disableLoadBar = False, columns_use = encoders)  # application of specific multicolumn setting IDF
        idfDicionary = cane.idfDictionary(Original = _df_train, Transformed = IDF_filter, columns_use = encoders) #, targetColumn=target)
        
        for col in encoders:
            _df_test[col] = (_df_test[col]
                             .map(idfDicionary[col])                  
                             .fillna(max(idfDicionary[col].values()))) #self.idf_dict -> dici IDF
                            
        for elemento in encoders:
            _df_train[elemento]=IDF_filter[elemento]

        _df_train=transform_dataset(df_train_,_df_train[encoders])
        _df_test=transform_dataset(df_test_,_df_test[encoders])
    else:
        print("###### No Categorical Columns ######")

    return _df_train,_df_test

def encode_label(df_train: pd.DataFrame, df_test: pd.DataFrame, target:str) -> tuple:
    
    """
    The encode_label function takes a DataFrame and encodes the categorical columns.
    It returns two DataFrames, one with the encoded data and another with the original data.
    
    :param df_train:pd.DataFrame: Pass the training dataset
    :param df_test:pd.DataFrame: Encode the test dataset
    :param target:str: Specify the target column
    """
    
    df_train=reset_index_DF(df_train)
    df_test=reset_index_DF(df_test)
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()
    _df_train = df_train_.copy()
    _df_test = df_test_.copy()
    
    _encoders = []
    encoders=categorical_columns(_df_train,target)
    
    if len(encoders)>0:

        for enc in encoders:
            values = _df_train[enc].unique()
            _values = pd.Series(values).dropna().reset_index(drop=True)
            _values.index += 1
            _values.name = enc
            _encoders.append(_values)
            dict_values = dict(map(reversed, _values.to_dict().items()))
    
            _df_train[enc] = _df_train[enc].map(dict_values.get, na_action="ignore")
            values_test = _df_test[enc]
            values_test = pd.Series(
                values_test[~values_test.isin(_values)].unique()
            ).dropna()
            _values = pd.concat([_values, values_test], ignore_index=True)
            _values.index += 1
            _values.name = enc
            dict_values = dict(map(reversed, _values.to_dict().items()))
            _df_test[enc] = _df_test[enc].map(dict_values.get)
        encoding_dict = pd.concat(_encoders, axis=1)

        _df_train=transform_dataset(df_train_,_df_train[encoders])
        _df_test=transform_dataset(df_test_,_df_test[encoders])
    else:
        print("###### No Categorical Columns ######")

    return _df_train, _df_test

def encode_standard(df_train: pd.DataFrame, df_test: pd.DataFrame, target:str) -> tuple:
    
    """
    The encode_standard function takes in a training and test set, as well as the target variable. 
    It then encodes all of the categorical variables using one-hot encoding. 
    The function returns two transformed datasets: df_train_ and df_test_. 
    
    :param df_train:pd.DataFrame: Pass the training dataset
    :param df_test:pd.DataFrame: Make sure that the test dataframe is not changed
    :param target:str: Select the target column from the training dataset
    """
    
    df_train=reset_index_DF(df_train)
    df_test=reset_index_DF(df_test)
    
    encoders=numerical_columns(df_train,target)
    
    if len(encoders)>0:
        
        df_train_ = df_train[encoders].copy()
        df_test_ = df_test[encoders].copy() 


        scaler = StandardScaler()
        scaler.fit(df_train_)

        df=pd.DataFrame(scaler.transform(df_train_))
        df.columns=encoders
        df_train_=df

        df_t=pd.DataFrame(scaler.transform(df_test_))
        df_t.columns=encoders
        df_test_=df_t

        df_train_=transform_dataset(df_train,df_train_[encoders])
        df_test_=transform_dataset(df_test,df_test_[encoders])

    return df_train_,df_test_

def encode_minmax(df_train: pd.DataFrame, df_test: pd.DataFrame, target:str) -> tuple:
    
    """
    The encode_minmax function takes in a dataframe and the target column name. It then 
    applies the MinMaxScaler to all of the numerical columns, and returns two transformed 
    dataframes with those columns scaled. The first is for training, and the second is for testing.
    
    :param df_train:pd.DataFrame: Specify the train dataset
    :param df_test:pd.DataFrame: Make sure that the test dataset has the same columns as the train dataset
    :param target:str: Specify the target column name
    :return: A tuple with the encoded train and test datasets
    """
    
    df_train=reset_index_DF(df_train)
    df_test=reset_index_DF(df_test)
    
    encoders=numerical_columns(df_train,target)
    
    if len(encoders)>0:
        
        df_train_ = df_train[encoders].copy() 
        df_test_ = df_test[encoders].copy() 

        scaler = MinMaxScaler()
        scaler.fit(df_train_) 

        df=pd.DataFrame(scaler.transform(df_train_))
        df.columns=encoders
        df_train_=df

        df_t=pd.DataFrame(scaler.transform(df_test_))
        df_t.columns=encoders
        df_test_=df_t

        df_train_=transform_dataset(df_train,df_train_[encoders])
        df_test_=transform_dataset(df_test,df_test_[encoders])

    return df_train_,df_test_

#############################################################  Encodings Validation   ###############################################################################

def version1_Encoding(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The version1_Encoding function takes in a training and test dataframe, as well as the name of the target column. 
    It then encodes all categorical columns using one-hot encoding, and all numerical columns using standard scaling. 
    If there are any null values in either the training or test set, it will impute them with 0s for numerical values and 'MISSING' for categorical ones.

    :param df_train:pd.DataFrame: Specify the training dataset
    :param df_test:pd.DataFrame: Check if there are any missing values in the test data
    :param Target:str: Specify the target column name
    """

    df_train_ = df_train.copy()
    df_test_ = df_test.copy()
    _df_train = df_train_.copy()
    _df_test = df_test_.copy()
    
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 

    if len(list_num_cols)>0:
        _df_train,_df_test=encode_standard(_df_train,_df_test,Target)
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_idf(_df_train,_df_test,Target)

    if (_df_train.isnull().sum().sum() or _df_test.isnull().sum().sum()) != 0:
        _df_train_PRED, _df_test_PRED = simple_null_imputation(_df_train,_df_test,Target)
    else:
        _df_train_PRED, _df_test_PRED=_df_train.copy(), _df_test.copy()
    
    return _df_train_PRED, _df_test_PRED,_df_train,_df_test

def version2_Encoding(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The version2_Encoding function takes in a training and test dataframe, as well as the name of the target column.
    It then encodes all categorical columns using an idf encoding method, and all numerical columns using a minmax encoding method.
    If there are any null values in either the train or test set, it will impute these missing values with 0s for numerical columns 
    and 'N/A' for categorical columns.
    
    :param df_train:pd.DataFrame: Pass the training data set
    :param df_test:pd.DataFrame: Test the function with a dataset that has no missing values
    :param Target:str: Specify the target column name
    """
    
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()
    _df_train = df_train_.copy()
    _df_test = df_test_.copy()
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 

    if len(list_num_cols)>0:
        _df_train,_df_test=encode_minmax(_df_train,_df_test,Target)    
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_idf(_df_train,_df_test,Target)

    if (_df_train.isnull().sum().sum() or _df_test.isnull().sum().sum()) != 0:
        _df_train_PRED, _df_test_PRED = simple_null_imputation(_df_train,_df_test,Target)
    else:
        _df_train_PRED, _df_test_PRED=_df_train.copy(), _df_test.copy()
    
    return _df_train_PRED, _df_test_PRED,_df_train,_df_test

def version3_Encoding(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The version3_Encoding function takes in a dataframe and returns the following:
        1. _df_train_PRED, _df_test_PRED - The processed train/test dataframes with all categorical columns encoded 
                                          and null values imputed.
        2. _df_train, _df_test - The original train/test dataframes that were inputted into the function (for use later)
    
    :param df_train:pd.DataFrame: Specify the training dataset
    :param df_test:pd.DataFrame: Create the test set
    :param Target:str: Specify the target column name
    """
    
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()
    _df_train = df_train_.copy()
    _df_test = df_test_.copy()
    
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 
    
    if len(list_num_cols)>0:
        _df_train,_df_test=encode_standard(_df_train,_df_test,Target)
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_label(_df_train,_df_test,Target)
                                           
    if (_df_train.isnull().sum().sum() or _df_test.isnull().sum().sum()) != 0:
        _df_train_PRED, _df_test_PRED = simple_null_imputation(_df_train,_df_test,Target)
    else:
        _df_train_PRED, _df_test_PRED=_df_train.copy(), _df_test.copy()
    
    return _df_train_PRED, _df_test_PRED,_df_train,_df_test

def version4_Encoding(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The version4_Encoding function takes in a training and test dataframe, 
    and returns the following:
    _df_train_PRED, _df_test_PRED - The training and test dataframes with categorical variables encoded.
    The original input dataframes are returned as well (for version4 purposes). 
    
    :param df_train:pd.DataFrame: Pass the training dataframe
    :param df_test:pd.DataFrame: Create a copy of the test dataframe
    :param Target:str: Specify the target column name
    """
    
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()
    _df_train = df_train_.copy()
    _df_test = df_test_.copy()
    
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 
    
    if len(list_num_cols)>0:
        _df_train,_df_test=encode_minmax(_df_train,_df_test,Target)
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_label(_df_train,_df_test,Target)
    if (_df_train.isnull().sum().sum() or _df_test.isnull().sum().sum()) != 0:
        _df_train_PRED, _df_test_PRED = simple_null_imputation(_df_train,_df_test,Target)
    else:
        _df_train_PRED, _df_test_PRED=_df_train.copy(), _df_test.copy()
    
    return _df_train_PRED, _df_test_PRED,_df_train,_df_test

##########################################################  Select Best Encoding Method  ############################################################################

def Select_Encoding_Method(df_train:pd.DataFrame, 
                           df_test:pd.DataFrame, 
                           Target, pred_type:str,
                           eval_metric:str):
    """
    The Select_Encoding_Method function is used to select the best encoding method for a given dataset. 
    The function takes in 5 parameters:
        1) df_train: The training dataframe that will be used to train the model. This is a pandas dataframe object.
        2) df_test: The test dataframe that will be used to test the performance of our model on unseen data. This is a pandas 
                    dataframe object.    
        3) Target: A string indicating which column name represents the target variable in your dataset (y).  
        4) pred_type: A string indicating whether
    
    :param df_train:pd.DataFrame: Select the train dataframe
    :param df_test:pd.DataFrame: Test the model with a different dataset than the one used for training
    :param Target: Select the target variable
    :param pred_type:str: Select between regression or classification
    :param eval_metric:str: Select the evaluation metric used to compare the different encoding versions
    :return: The train and test datasets with the best encoding method
    """
    
    Train_=df_train.copy()
    Test_=df_test.copy()
    _Train_=Train_.copy()
    _Test_=Test_.copy()
    
    Train_V1_PRED,Test_V1_PRED,Train_V1, Test_V1=version1_Encoding(_Train_, _Test_,Target)
    Train_V2_PRED,Test_V2_PRED,Train_V2, Test_V2=version2_Encoding(_Train_, _Test_,Target)
    Train_V3_PRED,Test_V3_PRED,Train_V3, Test_V3=version3_Encoding(_Train_, _Test_,Target)
    Train_V4_PRED,Test_V4_PRED,Train_V4, Test_V4=version4_Encoding(_Train_, _Test_,Target)

    Pred_Performance_Version1=Predictive_Evaluation(Train_V1_PRED, Test_V1_PRED,Target,pred_type)
    Pred_Performance_Version2=Predictive_Evaluation(Train_V2_PRED, Test_V2_PRED,Target,pred_type)
    Pred_Performance_Version3=Predictive_Evaluation(Train_V3_PRED, Test_V3_PRED,Target,pred_type)
    Pred_Performance_Version4=Predictive_Evaluation(Train_V4_PRED, Test_V4_PRED,Target,pred_type)
    
    Perf_Version1=Pred_Performance_Version1[eval_metric].sum()
    Perf_Version2=Pred_Performance_Version2[eval_metric].sum()
    Perf_Version3=Pred_Performance_Version3[eval_metric].sum()
    Perf_Version4=Pred_Performance_Version4[eval_metric].sum()
    
    if pred_type=="Reg":
        print(" ")
        print("Predictive Performance Encoding Versions:")
        print("\n MAE Version 1: ", round(Perf_Version1, 5),
              "\n MAE Version 2: ", round(Perf_Version2, 5),
              "\n MAE Version 3: ", round(Perf_Version3, 5),
              "\n MAE Version 4: ", round(Perf_Version4, 5))
        metric="MAE"
    elif pred_type=="Class":
        print("Predictive Performance Encoding Versions:")
        print("\n AUC Version 1: ", round(Perf_Version1, 5),
              "\n AUC Version 2: ", round(Perf_Version2, 5),
              "\n AUC Version 3: ", round(Perf_Version3, 5),
              "\n AUC Version 4: ", round(Perf_Version4, 5))
        metric="AUC"
    
    List_Encoding=[Perf_Version1,Perf_Version2,Perf_Version3,Perf_Version4]
    List_Encoding.sort()
    Encoding_Method=""

    if List_Encoding[0]==Perf_Version1:
        Encoding_Method="Encoding Version 1"
        print("Encoding Version 1 was choosen with an ", metric, " of: ", round(Perf_Version1, 5))
        _Train_,_Test_=Train_V1, Test_V1
        
    elif List_Encoding[0]==Perf_Version2:
        Encoding_Method="Encoding Version 2"
        print("Encoding Version 2 was choosen with an ", metric, " of: ", round(Perf_Version2, 5))
        _Train_,_Test_=Train_V2, Test_V2        
    
    elif List_Encoding[0]==Perf_Version3:
        Encoding_Method="Encoding Version 3"
        print("Encoding Version 3 was choosen with an ", metric, " of: ", round(Perf_Version3, 5))
        _Train_,_Test_=Train_V3, Test_V3
    
    elif List_Encoding[0]==Perf_Version4:
        Encoding_Method="Encoding Version 4"
        print("Encoding Version 4 was choosen with an ", metric, " of: ", round(Perf_Version4, 5))
        _Train_,_Test_=Train_V4, Test_V4
    
    return _Train_,_Test_,Encoding_Method

############################################################# Encoding Transform Methods ########################################################################################

def encoding_v1(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The encoding_v1 function takes in a training and test dataframe, as well as the name of the target column.
    It returns two modified dataframes with all categorical columns encoded using frequency-inverse document 
    frequency encoding. The function will only encode those columns that are categorical (as determined by the 
    categorical_columns function). If there are numerical columns, they will be encoded using standard scaling.
    
    :param df_train:pd.DataFrame: Specify the training dataframe
    :param df_test:pd.DataFrame: Encode the test dataframe
    :param Target:str: Specify the target column
    """
    
    df_train,df_test=reset_index_DF(df_train),reset_index_DF(df_test)
    _df_train = df_train.copy()
    _df_test = df_test.copy()
    
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 

    if len(list_num_cols)>0:
        _df_train,_df_test=encode_standard(_df_train,_df_test,Target)    
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_idf(_df_train,_df_test,Target)

    return _df_train,_df_test

def encoding_v2(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The encoding_v2 function takes in a training and test dataframe, as well as the name of the target column.
    It then encodes all categorical columns using an idf encoding scheme. It also optionally encodes numerical columns using a minmax scaling scheme.
    The function returns two transformed dataframes: one for training and one for testing.
    
    :param df_train:pd.DataFrame: Pass the training dataframe
    :param df_test:pd.DataFrame: Encode the test dataframe
    :param Target:str: Specify the target column
    """
    
    df_train,df_test=reset_index_DF(df_train),reset_index_DF(df_test)
    _df_train = df_train.copy()
    _df_test = df_test.copy()
    
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 

    if len(list_num_cols)>0:
        _df_train,_df_test=encode_minmax(_df_train,_df_test,list_num_cols)
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_idf(_df_train,_df_test,Target)

    return _df_train,_df_test

def encoding_v3(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The encoding_v3 function takes in a training and test dataframe, as well as the target column name. 
    It then encodes the categorical columns using Label Encoding and Standard Encoding. The function returns two encoded dataframes.
    
    :param df_train:pd.DataFrame: Specify the training dataframe
    :param df_test:pd.DataFrame: Check if the encoding is done correctly
    :param Target:str: Specify the target column name
    """
    
    df_train,df_test=reset_index_DF(df_train),reset_index_DF(df_test)
    _df_train = df_train.copy()
    _df_test = df_test.copy()
    
    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 
    
    if len(list_num_cols)>0:
        _df_train,_df_test=encode_standard(_df_train,_df_test,Target)
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_label(_df_train,_df_test,Target)
    
    return _df_train,_df_test

def encoding_v4(df_train:pd.DataFrame, df_test:pd.DataFrame, Target:str):
    
    """
    The encoding_v4 function takes in a training and test dataframe, as well as the name of the target column.
    It then encodes all categorical columns using Label Encoding, and all numerical columns using MinMax Scaling.
    The function returns two transformed dataframes: one for training and one for testing.
    
    :param df_train:pd.DataFrame: Specify the training dataset
    :param df_test:pd.DataFrame: Get the test dataframe
    :param Target:str: Specify the target column name
    """
    
    df_train,df_test=reset_index_DF(df_train),reset_index_DF(df_test)
    _df_train = df_train.copy()
    _df_test = df_test.copy()

    list_num_cols=numerical_columns(_df_train,Target)  
    list_cat_cols=categorical_columns(_df_train,Target) 

    if len(list_num_cols)>0:
        _df_train,_df_test=encode_minmax(_df_train,_df_test,Target)
    if len(list_cat_cols)>0:
        _df_train,_df_test=encode_label(_df_train,_df_test,Target)
    
    return _df_train,_df_test

########################################################### Feature Selection ##############################################################

###################################  H2O Feature Selection ######################################

def feature_selection_h2o(Dataset:pd.DataFrame, target:str, total_vi :float=0.98, h2o_fs_models:int =7, encoding_fs:bool=True):
    """
    The feature_selection_h2o function is used to select the most important input variables for a given model.
    The function takes as input: 
        - Dataset: A pandas dataframe with all the columns of your dataset, including target variable and features. 
        - Target: The name of your target variable (string). 
        - Total_vi : The total relative importance percentage you want to keep in your dataset (float). It should be between 0.5 and 1.0 . Default value is 0.98 .  
         If you want to use default value, just leave it blank or type None .
    
         Optional
    
    :param Dataset:pd.DataFrame: Input the dataset
    :param target:str: Define the target column of the dataset
    :param total_vi:float=0.98: Define the minimum percentage of relative importance that will be used to select the input columns
    :param h2o_fs_models:int=7: Define the number of models to be used in the h2o automl process
    :param encoding_fs:bool=True: Encode categorical variables
    """
    
    assert total_vi>=0.5 and total_vi<=1 , "total_vi value should be in [0.5,1] interval"
    assert h2o_fs_models>=1 and h2o_fs_models<=50 , "h2o_fs_models value should be in [0,50] interval"
    
    Train_=Dataset.copy()
    Train=Train_.copy()
    
    if encoding_fs==True:
        le =LabelEncoder()
        cols=categorical_columns(Train_,target)   
        Train_=Train_[cols]
        Train_ = Train_.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
        Train=transform_dataset(Train,Train_)
    elif encoding_fs==False:
        print("    Encoding method was not applied    ")

    Train=remove_columns_by_nulls(Train, 99) 
    
    Input_Cols= list(Train.columns)
    Input_Cols.remove(target)
    
    Train_H20=h2o.H2OFrame(Train)
    
    aml = H2OAutoML(max_models=h2o_fs_models,nfolds=3 , seed=1, exclude_algos = ["GLM", "DeepLearning", "StackedEnsemble"],sort_metric = "AUTO")
    aml.train(x=Input_Cols,y=target,training_frame=Train_H20)
    leaderboards = aml.leaderboard
    leaderboards_df= leaderboards.as_data_frame()
    print(leaderboards_df)
    
    List_Id_Model=[]
    
    for row in leaderboards_df["model_id"]:  
        List_Id_Model.append(row)
     
    print("Selected Leaderboard Model: ", List_Id_Model[0])
        
    m = h2o.get_model(List_Id_Model[0])
    Importancia_Total=m.varimp(use_pandas=True)
        
    Variable_Importance_=Importancia_Total.copy()
        
    n=0.015
    Variable_Importance_List_DF = Variable_Importance_[Variable_Importance_['percentage'] > n]
    Soma_Variable_Importance=Variable_Importance_List_DF['percentage'].sum()
    for iteration in range(0,10):
        print("Approximate minimum value of Relative Percentage:",n)
            
        if Soma_Variable_Importance<=total_vi:
            Variable_Importance_List_DF = Variable_Importance_[Variable_Importance_['percentage'] > n]
            n=n*0.5
            Soma_Variable_Importance=Variable_Importance_List_DF['percentage'].sum()
        elif Soma_Variable_Importance>total_vi:
            break
    Selected_Columns=[]
    for rows in Variable_Importance_List_DF["variable"]:
        Selected_Columns.append(rows)   
    print("Total amount of selected input columns: ", len(Selected_Columns))
    print("Total relative importance percentage of the selected columns: ", round(Soma_Variable_Importance*100, 4), "%")
    if len(Selected_Columns)>=5:
        List_Top_5_Cols=Selected_Columns[0:5]
        print("Top 5 Most Important Input Columns: ", List_Top_5_Cols)
    Selected_Columns.append(target)
    Selected_Importance=Variable_Importance_List_DF.copy()    

    return Selected_Columns, Selected_Importance

###############################################  VIF #########################################

def calc_vif(X):
    
    """
    The calc_vif function calculates the variance inflation factor for each variable in a dataframe.
    It returns a pandas DataFrame with two columns: variables and VIF. The variables column contains 
    the names of the independent variables, while the VIF column contains their respective values.
    
    :param X: Specify the dataframe that contains all the independent variables
    :return: A dataframe with the variables and their respective vifs
    """
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.sort_values(["VIF"], ascending=False)
    return vif

def feature_selection_VIF(Dataset:pd.DataFrame, target:str, VIF:float=10.0):
    
    """
    The feature_selection_VIF function takes a pandas dataframe and the name of the target column as input.
    It then calculates VIF for all columns in the dataset and returns a list of selected features based on 
    VIF value. The function also returns VIF_Dataset which is used to plot graph between Variance Inflation Factor 
    and independent variables.
    
    :param Dataset:pd.DataFrame: Pass the dataset
    :param target:str: Specify the target variable
    :param VIF:float=10.0: Set the threshold for vif value
    """
    
    assert VIF>=3 and VIF<=30 , "VIF value should be in [3,30] interval"
    
    Input_Cols= list(Dataset.columns)
    Input_Cols.remove(target)
    Dataset_=Dataset[Input_Cols]
    VIF_Dataset=calc_vif(Dataset_)
    Selected_Columns=Input_Cols
    for line in range(0,len(VIF_Dataset['VIF'])):
        if VIF_Dataset["VIF"].loc[VIF_Dataset['VIF'].idxmax()]>=VIF:
            VIF_Dataset.drop(VIF_Dataset["variables"].loc[VIF_Dataset['VIF']==VIF_Dataset["VIF"].max()].index, inplace=True)
            Selected_Columns=[]
            for rows in VIF_Dataset["variables"]:
                Selected_Columns.append(rows)
        Dataset_=Dataset_[Selected_Columns]
        VIF_Dataset=calc_vif(Dataset_)
    Selected_Columns.append(target)

    return Selected_Columns,VIF_Dataset

def vif_performance_selection(train:pd.DataFrame, 
                              test:pd.DataFrame, 
                              target:str, 
                              vif_ratio:float=10.0):
    
    """
    The vif_performance_selection function is used to select the best features from a dataset. 
    It uses the VIF (Variance Inflation Factor) method to remove columns with high multicollinearity. 
    The function takes in 5 parameters: train, test, target, vif_ratio and pred_type. The train parameter is a pandas dataframe of training data containing all the features and target variable(s). The test parameter is a pandas dataframe of testing/validation data containing all the features and target variable(s). The target parameter specifies which column in your dataset contains your dependent or response variable that you are trying
    
    :param train:pd.DataFrame: Pass the training dataset
    :param test:pd.DataFrame: Test the performance of the model on a different dataset
    :param target:str: Specify the target column name
    :param vif_ratio:float=10.0: Set the threshold for vif (variance inflation factor) value
    :return: The train and test datasets after feature selection
    """
    
    assert vif_ratio>=3 and vif_ratio<=30 , "vif_ratio value should be in [3,30] interval"
    
    train_=train.copy()
    test_=test.copy()
    _train_=train_.copy()
    _test_=test_.copy()
    
    pred_type, eval_metric=target_type(_train_, target)
    
    if pred_type=="Class":
        metric="AUC"
    elif pred_type=="Reg":
        metric="MAE"
    
    Selected_Columns=list(train_.columns)
    
    Pred_Performance_Default = Predictive_Evaluation(_train_, _test_,target,pred_type)
    Default_Performance=Pred_Performance_Default[eval_metric][0]
    
    _train__=train_.copy()
    _test__=test_.copy()
    try:
        Selected_Columns_VIF,VIF_Dataset=feature_selection_VIF(_train__,target,vif_ratio)
        print("Number of Selected VIF Columns: ", len(Selected_Columns_VIF), 
              "\n Removed Columns with VIF (Feature Selection - VIF):", len(Selected_Columns) - len(Selected_Columns_VIF), 
              "\n Selected Columns:", Selected_Columns_VIF)
        _train__=_train__[Selected_Columns_VIF]
        _test__=_test__[Selected_Columns_VIF]
    except Exception:
        print("traceback.format_exc: ", traceback.format_exc())

        
    Pred_Performance_VIF = Predictive_Evaluation(_train__, _test__,target,pred_type)
    Perf_Default_VIF=Pred_Performance_VIF[eval_metric][0]
    print("   ")
    print("Default Performance:",Default_Performance)
    print("Performance Default VIF:",Perf_Default_VIF)
    
    if Perf_Default_VIF<=Default_Performance:
        print("The VIF filtering method was applied    ")
        _train_=_train_[Selected_Columns_VIF]
        _test_=_test_[Selected_Columns_VIF]
    else:
        print("The VIF filtering method was not applied    ")
    return _train_, _test_
   
########################################################### Metrics ########################################################################

def metrics_regression(y_real, y_pred): 
    
    """
    The metrics_regression function calculates the metrics of a regression model.
    It takes as input two arrays: y_real and y_pred, which are the real values and 
    the predicted values of a target variable respectively. It returns a dictionary 
    with all the metrics.
    
    :param y_real: Store the real values of the target variable
    :param y_pred: Predict the values of y based on the model
    :return: A dictionary with the metrics of regression
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
    
    """
    The metrics_classification function takes in two parameters: y_true and y_pred. 
    It returns a dictionary of metrics for the model's performance on the test set.
    
    :param y_true: Pass the actual labels of the data
    :param y_pred: Store the predicted values
    :return: A dictionary containing the accuracy, precision and f-score
    """

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
    
    """
    The metrics_binary_classification function takes in two parameters: y_true and y_pred. 
    It returns a dictionary of metrics for the binary classification task.

    :param y_true: Specify the true class labels of the input samples
    :param y_pred: Pass the predicted values of the target variable
    :return: A dictionary of the metrics that are used to evaluate binary classification problems
    """

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

def Reg_RandomForest_Prediction(Train, Test, Target):
    
    """
    The Reg_RandomForest_Prediction function takes in a training and test set, as well as the target variable. 
    It then trains a Random Forest Regressor on the training data and predicts on the test data. It returns a DataFrame with performance metrics for each estimator.

    :param Train: Specify the training data
    :param Test: Define the test dataframe
    :param Target: Define the target column
    :return: A dataframe with the performance of each number of estimators
    """

    Selected_Cols= list(Train.columns)
    Selected_Cols.remove(Target)
    Selected_Cols.append(Target) 
    Train=Train[Selected_Cols]
    Test=Test[Selected_Cols]   
    
    X_train = Train.iloc[:, 0:(len(Selected_Cols)-1)].values
    X_test = Test.iloc[:, 0:(len(Selected_Cols)-1)].values
    y_train = Train.iloc[:, (len(Selected_Cols)-1)].values
    y_test = Test.iloc[:, (len(Selected_Cols)-1)].values
    
    List_Estimators,List_Algos=[100,250,500],[]
    
    for Estimators in List_Estimators:
        
        regressor_RF = RandomForestRegressor(n_estimators=Estimators, random_state=42)
        regressor_RF.fit(X_train, y_train)
        y_pred = regressor_RF.predict(X_test)
        RF_Performance=pd.DataFrame(metrics_regression(y_test,y_pred),index=[0])
        RF_Performance[["Estimators"]]=Estimators
        List_Algos.append(RF_Performance)
    RF_Dataframe=pd.concat(List_Algos)
    
    return RF_Dataframe

def Reg_ExtraTrees_Prediction(Train, Test, Target):
    
    """
    The Reg_ExtraTrees_Prediction function takes in a Train and Test dataframe, 
    and the Target column name as inputs. It then performs an ExtraTreesRegressor model on the train set, 
    and returns a dataframe with metrics of the test set.

    :param Train: Train the model
    :param Test: Test the model with a different dataset
    :param Target: Specify the target column in the train and test dataframes
    :return: A dataframe with the performance of each algorithm for every number of estimators
    """

    Selected_Cols= list(Train.columns)
    Selected_Cols.remove(Target)
    Selected_Cols.append(Target) 
    Train=Train[Selected_Cols]
    Test=Test[Selected_Cols]  
    
    X_train = Train.iloc[:, 0:(len(Selected_Cols)-1)].values
    X_test = Test.iloc[:, 0:(len(Selected_Cols)-1)].values
    y_train = Train.iloc[:, (len(Selected_Cols)-1)].values
    y_test = Test.iloc[:, (len(Selected_Cols)-1)].values
    
    List_Estimators,List_Algos=[100,250,500],[]
    
    for Estimators in List_Estimators:
        
        Reg_ET = ExtraTreesRegressor(n_estimators=Estimators, random_state=42)
        Reg_ET.fit(X_train, y_train)
        y_pred = Reg_ET.predict(X_test)
        ET_Performance=pd.DataFrame(metrics_regression(y_test,y_pred),index=[0])
        ET_Performance[["Estimators"]]=Estimators
        List_Algos.append(ET_Performance)
    
    ET_Dataframe=pd.concat(List_Algos) 
    return ET_Dataframe

def Class_RandomForest_Prediction(Train, Test, Target):
    
    """
    The Class_RandomForest_Prediction function takes in a Train and Test dataframe, 
    and the Target column name as inputs. It then performs Random Forest Classification on the training set, 
    and returns a dataframe with performance metrics for each estimator value.

    :param Train: Pass the train dataset
    :param Test: Define if the function is used to train and test on the training set or to make predictions on a test set
    :param Target: Specify the target column in the train and test dataframes
    :return: A dataframe that contains the accuracy, precision, recall and f-score for different values of estimators
    """

    Selected_Cols= list(Train.columns)
    Selected_Cols.remove(Target)
    Selected_Cols.append(Target) 
    Train=Train[Selected_Cols]
    Test=Test[Selected_Cols]   
    
    List_Estimators,List_Algos=[100,250,500],[]
    
    for Estimators in List_Estimators:
        
        X_train = Train.iloc[:, 0:(len(Selected_Cols)-1)].values
        X_test = Test.iloc[:, 0:(len(Selected_Cols)-1)].values
        y_train = Train.iloc[:, (len(Selected_Cols)-1)].values
        y_test = Test.iloc[:, (len(Selected_Cols)-1)].values
        
        classifier_RF = RandomForestClassifier(n_estimators=Estimators, random_state=42)
        classifier_RF.fit(X_train, y_train)
        y_pred = classifier_RF.predict(X_test)
        RF_Performance=pd.DataFrame(metrics_classification(y_test,y_pred),index=[0])
        RF_Performance[["Estimators"]]=Estimators
        List_Algos.append(RF_Performance)
    
    RF_Dataframe=pd.concat(List_Algos)
    
    return RF_Dataframe

def Class_ExtraTrees_Prediction(Train, Test, Target):
    
    """
    The Class_ExtraTrees_Prediction function is used to predict the target variable for a given test dataset. 
    The function takes in three parameters: Train, Test and Target. The Train parameter is the training dataframe 
    that contains both features and target variable(the feature that we want to predict). The Test parameter is the test dataframe that contains all features except for the target variable which we are trying to predict using our trained model. Finally, Target is a string containing name of our target feature.

    :param Train: Specify the training dataset
    :param Test: Determine if the model is trained on a test set or not
    :param Target: Specify the target column in the train and test dataframes
    :return: A dataframe with the performance of each number of estimators
    """

    Selected_Cols= list(Train.columns)
    Selected_Cols.remove(Target)
    Selected_Cols.append(Target) 
    Train=Train[Selected_Cols]
    Test=Test[Selected_Cols]  
    
    List_Estimators,List_Algos=[100,250,500],[]
    
    for Estimators in List_Estimators:
        
        X_train = Train.iloc[:, 0:(len(Selected_Cols)-1)].values
        X_test = Test.iloc[:, 0:(len(Selected_Cols)-1)].values
        y_train = Train.iloc[:, (len(Selected_Cols)-1)].values
        y_test = Test.iloc[:, (len(Selected_Cols)-1)].values
        
        classifier_ET = ExtraTreesClassifier(n_estimators=Estimators, random_state=42)
        classifier_ET.fit(X_train, y_train)
        y_pred = classifier_ET.predict(X_test)
        ET_Performance=pd.DataFrame(metrics_classification(y_test,y_pred),index=[0])
        ET_Performance[["Estimators"]]=Estimators
        List_Algos.append(ET_Performance)
    
    ET_Dataframe=pd.concat(List_Algos) 
    
    return ET_Dataframe

###########################################################    Predictive Evaluation   ######################################################

def Predictive_Evaluation(Train_DF,Test_DF, 
                          Target, 
                          pred_type:str):
    """
    The Predictive_Evaluation function takes in a training dataframe, testing dataframe, 
    a target column name and a prediction type (Class or Reg). It then runs the ExtraTrees 
    and RandomForest models on both datasets. The results are concatenated together and sorted by metric. 
    The top two rows of the resulting table are averaged to produce one final row of metrics.
    
    :param Train_DF: Specify the training dataframe
    :param Test_DF: Test the model on a dataset that is not used for training
    :param Target: Define the target variable for the model
    :param pred_type:str: Specify whether the function is used for classifcation or regression
    :return: The mean absolute error for the extratrees and randomforest models
    """
    
    train=Train_DF.copy()
    test=Test_DF.copy()
    
    if pred_type=="Class":
        metric='Accuracy'
        a=Class_ExtraTrees_Prediction(train,test,Target)
        b=Class_RandomForest_Prediction(train,test,Target)
    elif pred_type=="Reg":
        metric='Mean Absolute Error'
        a=Reg_ExtraTrees_Prediction(train,test,Target)
        b=Reg_RandomForest_Prediction(train,test,Target)

    x=pd.concat([a,b]) 
    x=x.sort_values(metric, ascending=True)
    del x['Estimators']
    
    y,z=x.iloc[:1,:],x.iloc[1:2,:]
    Metrics_Final=(y+z)/2

    return Metrics_Final

#############################################################################################################################################
###########################################################    Atlantic Pipeline   ##########################################################
#############################################################################################################################################

def atlantic_data_processing(Dataset:pd.DataFrame,
                             Target:str, 
                             Split_Racio:float,
                             total_vi:float=0.98,
                             h2o_fs_models:int =7,
                             encoding_fs:bool=True,
                             vif_ratio:float=10.0):
    
    """
    The atlantic_data_processing function is used to preprocess the input dataframe. 
    It is composed of several steps:
        1) Remove columns with more than 99% of null values;
        2) Remove Target Null Values;
        3) Datetime Feature Engineering
        4) Feature Selection by Variance H2O AutoML Variable Importance; 
        5) Encoding Method Selection;
        6) NULL SUBSTITUTION + ENCODING APLICATION;
        7) Feature Selection by Variance Inflation Factor (VIF); 
    
       The function returns a DataFrame_Final, Train and Test DataFrames transformed with the best performance selected preprocessing methods.
    
    :param Dataset:pd.DataFrame: Pass the dataset to be processed
    :param Target:str: Define the target column
    :param Split_Racio:float: Define the size of the test set
    :param total_vi:float=0.98: Select the most relevant features
    :param h2o_fs_models:int=7: Select the number of models used in the feature selection process
    :param encoding_fs:bool=True: Select the encoding method
    :param vif_ratio:float=10.0: Control the vif ratio
    """
    
    Dataframe_=Dataset.copy()
    Dataset_=Dataframe_.copy()

############################## Validation Dataframe ##############################

    Dataset_=remove_columns_by_nulls(Dataset_, 99.99)
    Selected_Cols= list(Dataset_.columns)
    Selected_Cols.remove(Target)
    Selected_Cols.append(Target) 
    Dataset_=Dataset_[Selected_Cols] ## Target -> Last Column Index

    train, test= split_dataset(Dataset_,Split_Racio)

    train_=train.copy()
    Train=train_.copy()
    test_=test.copy()
    Test=test.copy()

    Train=del_nulls_target(Train,Target) ## Delete Target Null Values
    Test=del_nulls_target(Test,Target) ## Delete Target Null Values
    
    Pred_type, Eval_metric=target_type(Dataset_, Target) ## Prediction Contextualization
    
############################## Feature Engineering Date Column ##############################

    Train=engin_date(Train)
    Test=engin_date(Test)

############################## Feature Selection ##############################
    
    Selected_Columns, Importancia_Selecionada =feature_selection_h2o(Train,Target,total_vi,h2o_fs_models,encoding_fs)
    print("Selected Columns:", Selected_Columns)

    Train=Train[Selected_Columns]
    Test=Test[Selected_Columns]

############################## Encoding Method Selection ##############################   

    Train_Back_Up,_Test_Back_Up,Encoding_Method=Select_Encoding_Method(Train,Test,Target,Pred_type,Eval_metric)

############################## NULL SUBSTITUTION + ENCODING APLICATION ##############################

    if (Train.isnull().sum().sum() or Test.isnull().sum().sum()) != 0:
        Train, Test,List_Imputation,Imputation_Method,Performance_Imputation_Algorithms=null_substitution_method(Train,Test,Target,Encoding_Method,Pred_type,Eval_metric)
    else:
        ## Encoding Method
        Imputation_Method="Undefined"
        if Encoding_Method=="Encoding Version 1":
            Train_PRED,Test_PRED,Train, Test=version1_Encoding(Train, Test,Target)
        elif Encoding_Method=="Encoding Version 2":
            Train_PRED,Test_PRED,Train, Test=version2_Encoding(Train, Test,Target)
        elif Encoding_Method=="Encoding Version 3":
            Train_PRED,Test_PRED,Train, Test=version3_Encoding(Train, Test,Target)
        elif Encoding_Method=="Encoding Version 4":
            Train_PRED,Test_PRED,Train, Test=version4_Encoding(Train, Test,Target)
            
        print("    ")   
        print("There are no missing values in the Input Data")    
        print("    ") 
        
############################## VARIANCE INFLATION FACTOR (VIF) APPLICATION ##############################
    
    Train, Test=vif_performance_selection(Train,Test,Target)

############################## Transformation Procediment ##############################

    Dataframe=Dataframe_.copy()
    Dataframe=del_nulls_target(Dataframe,Target) ## Delete Target Null Values 
    Dataframe=engin_date(Dataframe)
    Dataframe=Dataframe[list(Train.columns)]
    Train_DF,Test_DF=split_dataset(Dataframe,Split_Racio)
    Train_DF=Dataframe.copy()
    
    if Encoding_Method=="Encoding Version 1":
        Train_DF, Test_DF=encoding_v1(Train_DF, Test_DF,Target)
    elif Encoding_Method=="Encoding Version 2":
        Train_DF, Test_DF=encoding_v2(Train_DF, Test_DF,Target)
    elif Encoding_Method=="Encoding Version 3":
        Train_DF, Test_DF=encoding_v3(Train_DF, Test_DF,Target)
    elif Encoding_Method=="Encoding Version 4":
        Train_DF, Test_DF=encoding_v4(Train_DF, Test_DF,Target)

    if Imputation_Method=="Zero_Sub":  
        Train_DF,Test_DF=const_null_imputation(Train_DF, Test_DF,Target)
    elif Imputation_Method=="Simple":  
        Train_DF, Test_DF=simple_null_imputation(Train_DF, Test_DF,Target)
    elif Imputation_Method=="KNN":  
        Train_DF, Test_DF = knn_null_imputation(Train_DF, Test_DF,Target)
    elif Imputation_Method=="Iterative":
        Train_DF,Test_DF=iterative_null_imputation(Train_DF, Test_DF,Target)

    DataFrame_Final=reset_index_DF(Train_DF)

    return DataFrame_Final, Train, Test