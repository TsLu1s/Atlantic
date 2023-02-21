import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(Dataset:pd.DataFrame, Split_Racio:float):
    
    assert Split_Racio>=0.5 and Split_Racio<=0.95 , 'Split_Racio value should be in [0.5,0.95[ interval'
    
    train, test= train_test_split(Dataset, train_size=Split_Racio)

    return train,test

def target_type(Dataset:pd.DataFrame, target:str):  
    
    df=Dataset[[target]]
    reg_target,class_target=df.select_dtypes(include=['int','float']).columns.tolist(),df.select_dtypes(include=['object','category']).columns.tolist()
    if len(class_target)==1:
        pred_type='Class'
        eval_metric='Accuracy'
        
    elif len(reg_target)==1:
        pred_type='Reg'
        eval_metric='Mean Absolute Error'
        
    return pred_type, eval_metric

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

def slice_timestamp(Dataset:pd.DataFrame):
    
    df=Dataset.copy()
    datetime_cols=list_date=list(df.select_dtypes(include=['datetime','datetime64[ns]']))
    for date_col in datetime_cols:
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
        
    return Df

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


