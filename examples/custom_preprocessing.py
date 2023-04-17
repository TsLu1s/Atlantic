import atlantic as atl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console


url="https://raw.githubusercontent.com/TsLu1s/Atlantic/main/data/Fraudulent_Claim_Cars_class.csv"

data= pd.read_csv(url, encoding='latin', delimiter=',')

target="fraud"

data["claim_date"]=pd.to_datetime(data["claim_date"])
data=data[data[target].isnull()==False]
data=data.reset_index(drop=True)
data[target]=data[target].astype('category')

data.dtypes
data.isna().sum()

train,test = train_test_split(data, train_size=0.8)

#train_,test_=train.reset_index(drop=True), test.reset_index(drop=True)

#################################### ATLANTIC DATA PREPROCESSING #####################################

########################################################################
################# Date Time Feature Engineering ##################

train = atl.engin_date(train,drop=True)
test = atl.engin_date(test,drop=True)

########################################################################
################## Encoders ##################

## Option 1: MultiColumn LabelEncoder

le_fit=atl.fit_Label_Encoding(train,target)  
train_le=atl.transform_Label_Encoding(train,le_fit)
test_le=atl.transform_Label_Encoding(test,le_fit)
    
## Option 2: MultiColumn OneHotEncoder

ohe_fit=atl.fit_OneHot_Encoding(train,target,n_distinct=10)
train_2=atl.transform_OneHot_Encoding(train,ohe_fit)
test_2=atl.transform_OneHot_Encoding(test,ohe_fit)
    
## Option 3: MultiColumn IDF

idf_fit=atl.fit_IDF_Encoding(train,target)
train_3=atl.transform_IDF_Encoding(train,idf_fit)
test_3=atl.transform_IDF_Encoding(test,idf_fit)

########################################################################
################# Automated (Sklearn) Null Imputation [Only numeric features]

## Option 1: Simple Imputer
imputer=atl.fit_SimpleImp(train_le,
                          target=target,
                          strat='mean')

train_simple=atl.transform_SimpleImp(train_le,
                                     target=target,
                                     imputer=imputer)

test_simple=atl.transform_SimpleImp(test_le,
                                    target=target,
                                    imputer=imputer)

## Option 2: KNN Imputer
imputer_knn=atl.fit_KnnImp(train_le,
                           target=target,
                           neighbors=5)

train_knn=atl.transform_KnnImp(train_le,
                               target=target,
                               imputer=imputer_knn)

test_knn=atl.transform_KnnImp(test_le,
                              target=target,
                              imputer=imputer_knn)

## Option 3: Iterative Imputer
imputer_iter=atl.fit_IterImp(train_le,
                             target=target,
                             order='ascending')

train_iter=atl.transform_IterImp(train_le,
                                 target=target,
                                 imputer=imputer_iter)

test_iter=atl.transform_IterImp(test_le,
                                target=target,
                                imputer=imputer_iter)


########################################################################
################## Feature Selection ##################


## Option 1: H2O AutoML Feature Selection
sel_columns, h2o_imp = atl.feature_selection_h2o(dataset=data,       # dataset:pd.DataFrame ,target:str="Target_Column",
                                                 target=target,      #  total_vi:float [0.5,1], h2o_fs_models:int [1,50], encoding_fs:bool=True/False
                                                 total_vi=0.98,     
                                                 h2o_fs_models =7,
                                                 encoding_fs=True)

## Option 2: Variance Importance Factor Feature Selection
sel_columns, vif_imp = atl.feature_selection_vif(dataset=train_iter,   # dataset:pd.DataFrame, target:str="Target_Column",
                                                 target=target,  # VIF:float [3,30]
                                                 VIF=10.0)
