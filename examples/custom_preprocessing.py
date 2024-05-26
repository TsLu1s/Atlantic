from atlantic.processing.analysis import Analysis 
from atlantic.processing.scalers import (AutoMinMaxScaler, 
                                         AutoStandardScaler,
                                         AutoRobustScaler)
from atlantic.processing.encoders import (AutoLabelEncoder, 
                                          AutoIdfEncoder,
                                          AutoOneHotEncoder)
from atlantic.imputers.imputation import (AutoSimpleImputer, 
                                          AutoKNNImputer,
                                          AutoIterativeImputer)
from atlantic.feature_selection.selector import Selector  
import pandas as pd
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
train,test=train.reset_index(drop=True), test.reset_index(drop=True)

#################################### ATLANTIC DATA PREPROCESSING #####################################

########################################################################
################# Date Time Feature Engineering #################

an = Analysis(target=target)
train = an.engin_date(X=train,drop=True)
test = an.engin_date(X=test,drop=True)

########################################################################
################# Encoders #################

train_df,test_df=train.copy(),test.copy()

cat_cols=list(Analysis(target).cat_cols(X=train_df))

## Create Label Encoder
encoder = AutoLabelEncoder()
## Create IDF Encoder
encoder = AutoIdfEncoder()
## Create One-hot Encoder
encoder = AutoOneHotEncoder()

## Fit
encoder.fit(train_df[cat_cols])

# Transform the DataFrame using Label\IDF\One-hot Encoding
train_df=encoder.transform(X=train_df)
test_df=encoder.transform(X=test_df)

# Label Encoding : Perform an inverse transform to convert it back the categorical columns values
test_df = encoder.inverse_transform(X=test_df)

# IDF & One-hot Encoding : Perform an inverse transform to convert it back the categorical columns values
# Note: Only decodes the last transformed Dataframe
test_df = encoder.inverse_transform()

########################################################################
################# Scalers #################

train_df,test_df=train.copy(),test.copy()

num_cols = list(Analysis(target).num_cols(X=train_df))

### Standard Scaler
scaler = AutoStandardScaler()
### MinMax Scaler
scaler = AutoMinMaxScaler()
### Robust Scaler
scaler = AutoRobustScaler()

scaler.fit(X=train_df[num_cols])

train_df[num_cols] = scaler.transform(X=train_df[num_cols])

test_df[num_cols] = scaler.transform(X=test_df[num_cols])

test_df[num_cols] = scaler.inverse_transform(X=test_df[num_cols])


########################################################################
################# Automated Null Imputation [Only numeric features]

# Example usage of AutoSimpleImputer
simple_imputer = AutoSimpleImputer(strategy='mean')
simple_imputer.fit(train)  # Fit on the Train DataFrame
df_imputed = simple_imputer.transform(train.copy())  # Transform the Train DataFrame
df_imputed_test = simple_imputer.transform(test.copy()) # Transform the Test DataFrame

# Example usage of AutoKNNImputer
knn_imputer = AutoKNNImputer(n_neighbors=3,
                             weights="uniform")
knn_imputer.fit(train)  # Fit on the Train DataFrame
df_imputed = knn_imputer.transform(train.copy())  # Transform the Train DataFrame
df_imputed_test = knn_imputer.transform(test.copy()) # Transform the Test DataFrame

# Example usage of AutoIterativeImputer
iterative_imputer = AutoIterativeImputer(max_iter=10, 
                                         random_state=0, 
                                         initial_strategy="mean", 
                                         imputation_order="ascending")
iterative_imputer.fit(train)  # Fit on the Train DataFrame
df_imputed = iterative_imputer.transform(train.copy())  # Transform the Train DataFrame
df_imputed_test = iterative_imputer.transform(test.copy()) # Transform the Test DataFrame

train.select_dtypes(include=["float","int"]).isnull().sum()
df_imputed.select_dtypes(include=["float","int"]).isnull().sum()
test.select_dtypes(include=["float","int"]).isnull().sum()
df_imputed_test.select_dtypes(include=["float","int"]).isnull().sum()

########################################################################
################# Feature Selection #################

fs = Selector(X = train, target = target)
cols_vif = fs.feature_selection_vif(vif_threshold = 10.0)                           # X: Only numerical values allowed & No nans allowed
selected_cols, selected_importance = fs.feature_selection_h2o(relevance = 0.98,     # total_vi:float [0.5,1], h2o_fs_models:int [1,100]
                                                              h2o_fs_models = 7,    # encoding_fs:bool=True/False
                                                              encoding_fs = True)
