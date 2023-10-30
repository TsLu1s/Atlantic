######################################## REGRESSION ########################################

import atlantic as atl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

#source_data="https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees"

url="https://raw.githubusercontent.com/TsLu1s/Atlantic/main/data/Garments_Worker_Productivity_reg.csv"
data= pd.read_csv(url) 
data['date'] = pd.to_datetime(data['date'])

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True)

target_col="targeted_productivity"

### Fit Data Processing
atl = ATLpipeline(X=train,              # X:pd.DataFrame, target:str="Target_Column"
                  target=target_col)    

atl.fit_processing(split_ratio=0.75,   # split_ratio:float=0.75, relevance:float=0.98 [0.5,1]
                   relevance=0.99,     # h2o_fs_models:int [1,50], encoding_fs:bool=True\False
                   h2o_fs_models=7,    # vif_ratio:float=10.0 [3,30]
                   encoding_fs=True,
                   vif_ratio=10.0)

### Transform Data Processing
train = atl.data_processing(X=train)
test = atl.data_processing(X=test)


# Export Atlantic Preprocessing Metadata
import pickle 
output = open("fit_atl.pkl", 'wb')
pickle.dump(atl, output)