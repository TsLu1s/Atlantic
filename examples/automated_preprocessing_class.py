######################################## CLASSIFICATION ########################################

import atlantic.pipeline as Atlantic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

#source_data="https://www.kaggle.com/datasets/surekharamireddy/fraudulent-claim-on-cars-physical-damage"

url="https://raw.githubusercontent.com/TsLu1s/Atlantic/main/data/Fraudulent_Claim_Cars_class.csv"
data= pd.read_csv(url) # Dataframe Loading Example

target_col="fraud"
data[target_col]=data[target_col].astype('category')

data.dtypes
data.isna().sum()

train,test = train_test_split(data, train_size = 0.8)
test,future_data = train_test_split(test, train_size = 0.6)


# Resetting Index is Required
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

future_data.drop(columns=[target_col], inplace=True) # Drop Target

### Fit Data Processing
atl = Atlantic(X = train,              # X:pd.DataFrame, target:str="Target_Column"
               target = target_col)

atl.fit_processing(split_ratio = 0.75,   # split_ratio:float=0.75, relevance:float=0.99 [0.4,1]
                   relevance = 0.99,     # h2o_fs_models:int [1,100], encoding_fs:bool=True\False
                   h2o_fs_models = 7,    # vif_ratio:float=10.0 [3,30]
                   encoding_fs = True,
                   vif_ratio = 10.0)

### Transform Data Processing
train = atl.data_processing(X = train)
test = atl.data_processing(X = test)
future_data = atl.data_processing(X = future_data)


# Export Atlantic Preprocessing Metadata
import pickle 
output = open("fit_atl.pkl", 'wb')
pickle.dump(atl, output)

