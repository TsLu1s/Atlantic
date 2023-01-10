<br>
<p align="center">
  <h2 align="center"> Atlantic - Automated Preprocessing Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `Atlantic` project constitutes an comprehensive and objective approach to simplify and automate data processing through the integration and objectively validated application of various preprocessing mechanisms, ranging from feature engineering, automated feature selection, multiple encoding versions and null imputation methods. The optimization methodology of this framework follows a evaluation structured in tree based models ensembles.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed preprocessing procedures are applicable on any data table associated with Supervised Machine Learning scopes regardless of the properties or specifications of the Dataset features.

* Automated treatment of tabular data associated with predictive analysis: It implements a global and carefully validated tested data treatment based on the characteristics of each Dataset input columns, assuming its identification to the associated target column.

* Robustness and improvement of predictive results: The implementation of the `atlantic` automated data preprocessing function aims at improve predictive performance directly associated with the processing methods implemented based on the Dataset properties.  
    
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 
   
* [Python](https://www.python.org/downloads)
* [H2O.ai](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
* [Sklearn](https://scikit-learn.org/stable/)

    
## Framework Architecture <a name = "ta"></a>

<p align="center">
  <img src="https://i.ibb.co/7VDGscP/ATL-Final-Architecture.png" align="center" width="800" height="650" />
</p>    

## Where to get it <a name = "ta"></a>

Binary installer for the latest released version is available at the Python Package Index ([PyPI])(https://pypi.org/project/atlantic/). 

## Installation  

To install this package from Pypi repository run the following command:

```
pip install atlantic
```

# Usage Examples
    
## 1. Atlantic - Automated Preprocessing Pipeline

In order to be able to apply the automated preprocessing `atlantic` pipeline you need first to import the package. 
The following needed step is to load a dataset and define your to be predicted target column name into the variable `Target` and define split ratio for your Train and Test subsets.
You can customize the main function (customizable option) by altering the following running pipeline parameters:
* Split_Racio: Division ratio in wich the preprocessing methods will be evaluated within the loaded Dataset.
* total_vi: Minimal value of the total sum of relative variable\feature importance percentage selected in the "H2O AutoML feature selection" step.
* h2o_fs_models: Quantity of models generated for competition in step "H2O AutoML feature selection" to evaluate the relative importance of each feature (only leaderboard model will be selected for evaluation).
* encoding_fs: You can choose if you want to encond your features in order to reduce loading time in "H2O AutoML feature selection" step. If in "True" mode label encoding is applied to categorical features.
* vif_ratio: This value defines the minimal 'threshold' for Variance Inflation Factor filtering (default value=10)  
 
Importante Notes:
    
* Default predictive evaluation metric for regression contexts is MAE (Mean Absolute Error) and classification is AUC (Accuracy).
* Although functional, `Atlantic` data processing is not optimized for big data purposes yet.
* Major update is now available in **versions 1.0.1**
    
```py
    
import atlantic as atl
import pandas as pd   
    
data = pd.read_csv('csv_directory_path', encoding='latin', delimiter=',') # Dataframe Loading Example
   
train,test=atl.split_dataset(data,Split_Racio=0.8) 

### Fit Data Processing
    
# Simple Option
fit_atl = atl.fit_processing(Dataset=train,                # Dataset:pd.DataFrame, target:str="Name_Target_Column"
                             target="Name_Target_Column",  # Split_Racio:float=0.75 [0.5,0.95[ -> Recommended
                             Split_Racio=0.75)
    
# Customizable Option
fit_atl = atl.fit_processing(Dataset=train,                  # Dataset:pd.DataFrame, 
                             target="Name_Target_Column",    # target:str="Name_Target_Column"
                             Split_Racio=0.75,               # Split_Racio:float=0.75, total_vi:float=0.98 [0.5,1]
                             total_vi=0.98,                  # h2o_fs_models:int [1,50], encoding_fs:bool=True\False
                             h2o_fs_models=7,                # vif_ratio:float=10.0 [3,30]
                             encoding_fs=True,
                             vif_ratio=10.0)

### Transform Data Processing
    
train=atl.data_processing(train,
                          fit_atl)
test=atl.data_processing(test,
                         fit_atl)
    
```  

## 2. Atlantic - Preprocessing Data
    
### 2.1 Encoding Versions
 
There are multiple preprocessing functions available to direct use. This package provides upgrated encoding `LabelEncoder`, `OneHotEncoder` and [IDF](https://pypi.org/project/cane/) functions with an automatic multicolumn application, as are the scalers `Standard`, `MinMax` and `Robust`. 
 
* Note : `n_distinct` costumizable parameter in `OneHotEncoder` function constitutes the max limiter of distinct elements in columns, this meaning, columns with higher distinct values then 'n_distinct' will not be encoded.    

```py
import atlantic as atl
import pandas as pd 

train, test = atl.split_dataset(Dataset,Split_Racio=0.75) # Split Initial Dataframe
                                                          # Dataset:pd.DataFrame, Split_Racio:float
target = "Name_Target_Column" # -> target feature name
    
## Encoders
# MultiColumn LabelEncoder

le_fit=atl.fit_Label_Encoding(train,target)  
train=atl.transform_Label_Encoding(train,le_fit)
test=atl.transform_Label_Encoding(test,le_fit)
    
# MultiColumn OneHotEncoder

ohe_fit=atl.fit_OneHot_Encoding(train,target,n_distinct=10)
train=atl.transform_OneHot_Encoding(train,ohe_fit)
test=atl.transform_OneHot_Encoding(test,ohe_fit)
    
# MultiColumn IDF

idf_fit=atl.fit_IDF_Encoding(train,target)
train=atl.transform_OneHot_Encoding(train,idf_fit)
test=atl.transform_OneHot_Encoding(test,idf_fit)
        
    
## Scalers
# StandardScaler
    
scaler,num_cols=atl.fit_StandardScaler(train,target)
train[num_cols]=scaler.transform(train[num_cols])
test[num_cols]=scaler.transform(test[num_cols])  

# MinmaxScaler
    
scaler,num_cols=atl.fit_MinmaxScaler(train,target)
train[num_cols]=scaler.transform(train[num_cols])
test[num_cols]=scaler.transform(test[num_cols])     
    
# RobustScaler

scaler,num_cols=atl.fit_RobustScaler(train,target)
train[num_cols]=scaler.transform(train[num_cols])
test[num_cols]=scaler.transform(test[num_cols])  
    
```    
   
### 2.2 Feature Selection Methods

You can get filter your most valuable features from the dataset via this 2 feature selection methods:
    
* [H2O AutoML Feature Selection](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/variable-importance.html) - This function is based of how variable importance is calculated for tree-based models in H2Os AutoML and it can be customized by use of the following parameters: 
  * total_vi: Minimal value of the total sum of relative variable\feature importance percentage selected.
  * h2o_fs_models: Quantity of models generated for competition to evaluate the relative importance of each feature (only leaderboard model will be selected for evaluation).
  * encoding_fs: You can choose if you want to encond your features in order to reduce loading time in "H2O AutoML feature selection" step. If in "True" mode label encoding is applied to categorical features.
    
    
* [VIF Feature Selection (Variance Inflation Factor)](https://www.investopedia.com/terms/v/variance-inflation-factor.asp) - Variance inflation factor aims at measuring the amount of multicollinearity in a set of multiple regression variables or features, therefore for this filtering function to be applied all input variables need to be of numeric type. It can be customized by changing the column selection treshold (VIF:float) designated with a default value of 10.
    
    
```py    
    
selected_columns, h2o_importance = atl.feature_selection_h2o(Dataset, # Dataset:pd.DataFrame ,target:str="Name_Target_Column",
                                                             target,      #  total_vi:float [0.5,1], h2o_fs_models:int [1,50], encoding_fs:bool=True/False
                                                             total_vi=0.98,     
                                                             h2o_fs_models =7,
                                                             encoding_fs=True)


selected_columns, vif_importance = atl.feature_selection_VIF(Dataset, # Dataset:pd.DataFrame, target:str="Name_Target_Column",
                                                             target,  # VIF:float [3,30]
                                                             VIF=10.0)
```
    
### 2.3 Datetime Feature Engineering

The engin_date function converts and transforms columns of Datetime type into additional columns (Year, Day of the Year, Season, Month, Day of the month, Day of the week, Weekend, Hour, Minute) which will be added by association to the input dataset and subsequently deletes the original column if parameter drop=True.
    
    
```py   
    
dataset = atl.engin_date(Dataset,drop=False) # Dataset:pd.DataFrame, drop:bool
    
```

### 2.4 Predictive Performance Metrics

You can analyse the obtained predictive performance results by using the given bellow functions witch contains the most used metrics for each supervised predictive context.
    
    
```py  

reg_performance = pd.DataFrame(atl.metrics_regression(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
    
binary_class_Performance = pd.DataFrame(atl.metrics_binary_classification(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
    
multiclass_performance = pd.DataFrame(atl.metrics_classification(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
    
```

### 2.5 Extra Auxiliar Functions
    
The following functions were used in the development of this project.
    
```py  
    
## Data Preprocessing 
    
atl.reset_index_DF(Dataset:pd.DataFrame) # return dataset
    
atl.split_dataset(Dataset:pd.DataFrame,
                  Split_Racio:float) # return train, test
        
atl.num_cols(Dataset:pd.DataFrame,
             target:str) # return list_num_cols
    
atl.cat_cols(Dataset:pd.DataFrame,
             target:str) # return list_cat_cols 
    
atl.del_nulls_target(Dataset:pd.DataFrame,
                     target:str)  # return Dataset
    
atl.remove_columns_by_nulls(Dataset:pd.DataFrame,
                            percentage:int) # return dataset

## Simplified Null Imputation (Only numeric features)

imputer_knn=atl.fit_KnnImp(df:pd.DataFrame,
                           target:str,
                           neighbors:int=5)
df=atl.transform_KnnImp(df:pd.DataFrame,
                        target:str,
                        imputer=imputer_knn)

imputer_simple=atl.fit_SimpleImp(df:pd.DataFrame,
                                 target:str,
                                 strat:str='mean')
df=atl.transform_SimpleImp(df:pd.DataFrame,
                           target:str,
                           imputer=imputer_simple)
    
imputer_iter=atl.fit_IterImp(df:pd.DataFrame, 
                             target:str, 
                             order:str='ascending')
df=atl.transform_IterImp(df:pd.DataFrame,
                         target:str,
                         imputer=imputer_iter)
```   
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/Atlantic/blob/main/LICENSE) for more information.

## Contact 
 
[Luís Santos - LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)   
