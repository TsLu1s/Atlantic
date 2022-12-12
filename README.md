<br>
<p align="center">
  <h2 align="center"> Atlantic - Automated Preprocessing Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `Atlantic` project constitutes an comprehensive and objective approach to simplify and automate data processing through the integration and objectively validated application of various preprocessing mechanisms, ranging from feature engineering, automated feature selection, multiple encoding versions and null imputation methods. The optimization methodology of this framework follows a evaluation structured in tree-based models by the implemention of Random Forest and Extra Trees ensembles.

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
  <img src="https://i.ibb.co/wgfxFCc/ATL-Architecture-Final.png" align="center" width="800" height="600" />
  
</p>  

## Where to get it <a name = "ta"></a>
    
The source code is currently hosted on GitHub at: https://github.com/TsLu1s/Atlantic

Binary installer for the latest released version are available at the Python Package Index (PyPI).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install atlantic
```

# Usage Examples
    
## 1. Atlantic - Automated Preprocessing Pipeline

In order to be able to apply the automated preprocessing `atlantic` pipeline you need first to import the package. 
The following needed step is to load a dataset and define your to be predicted target column name into the variable `Target` and define split ratio for your Train and Test subsets (default value=0.75).
You can also customize the main function further (customizable option) by altering the following running pipeline parameters:
* total_vi: Minimal value of the total sum of relative variable\feature importance percentage selected in the "H2O AutoML feature selection" step.
* h2o_fs_models: Quantity of models generated for competition in step "H2O AutoML feature selection" to evaluate the relative importance of each feature (only leaderboard model will be selected for evaluation).
* encoding_fs: You can choose if you want to encond your features in order to reduce loading time in "H2O AutoML feature selection" step. If in "True" mode label encoding is applied to categorical features.
* vif_ratio: This value defines the minimal 'threshold' for Variance Inflation Factor filtering (default value=10)  
 
Importante Notes:
    
* Default predictive evaluation metric for regression contexts is MAE (mean absolute error) and classification is AUC (Accuracy).
* In order to avoid data leakage only Train\Test processed returned Datasets should be used for predictive analysis, `Processed_Dataset` variable should only be used for observation purposes.
    
```py
    
import atlantic as atl
import pandas as pd 

data = pd.read_csv('csv_directory_path', encoding='latin', delimiter=',') # Dataframe Loading Example
#target = "Name_Target_Column" # -> Define Target Feature to Predict

   
# Simple Option
processed_dataset,train,test = atl.atlantic_data_processing(Dataset=data,                 # Dataset:pd.DataFrame, target:str="Name_Target_Column"
                                                            target="Name_Target_Column",  # Split_Racio:float=0.75 [0.5,0.95[ -> Recommended
                                                            Split_Racio=0.75)
    
# Customizable Option
processed_dataset,train,test = atl.atlantic_data_processing(Dataset=data,                   # Dataset:pd.DataFrame, 
                                                            target="Name_Target_Column",    # target:str="Name_Target_Column"
                                                            Split_Racio=0.75,               # Split_Racio:float=0.75, total_vi:float=0.98 [0.5,1]
                                                            total_vi=0.98,                  # h2o_fs_models:int [1,50],  encoding_fs:bool=True\False
                                                            h2o_fs_models=7,                # vif_ratio:float=10.0 [3,30]
                                                            encoding_fs=True,
                                                            vif_ratio=10.0)
```  

## 2. Atlantic - Preprocessing Data
    
### 2.1 Encoding Versions
 
There are 4 different main encoding versions available to direct use. This were generated through the combination of the following distinct preprocessing methods:

* [Sklearn MinMaxScaler](https://scikit-learn.org/stable/modules/preprocessing.html) 
* [Sklearn StandardScaler](https://scikit-learn.org/stable/modules/preprocessing.html)
* [Sklearn LabelEncoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
* [Inverse Document Frequency (IDF)](https://pypi.org/project/cane/) 
    

```py

    
train, test = atl.split_dataset(Dataset,Split_Racio=0.75) # Split Initial Dataframe
                                                          # Dataset:pd.DataFrame, Split_Racio:float

train, test = atl.encoding_v1(train,test,target) ## Implements IDF to Categorical Features, StandardScaler to Numeric Features
train, test = atl.encoding_v2(train,test,target) ## Implements IDF to Categorical Features, MinMaxScaler to Numeric Features
train, test = atl.encoding_v3(train,test,target) ## Implements LabelEncoding to Categorical Features, StandardScaler to Numeric Features
train, test = atl.encoding_v4(train,test,target) ## Implements LabelEncoding to Categorical Features, MinMaxScaler to Numeric Features

# train:pd.DataFrame, test:pd.DataFrame, target:str="Name_Target_Column"        
    
```    
   
### 2.2 Feature Selection Methods

You can get filter your most valuable features from the dataset via this 2 feature selection methods:
    
* [H2O AutoML Feature Selection](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/variable-importance.html) - This function is based of how variable importance is calculated for tree-based models in H2Os AutoML and it can be customized by use of the following parameters: 
  * total_vi: Minimal value of the total sum of relative variable\feature importance percentage selected.
  * h2o_fs_models: Quantity of models generated for competition to evaluate the relative importance of each feature (only leaderboard model will be selected for evaluation).
  * encoding_fs: You can choose if you want to encond your features in order to reduce loading time in "H2O AutoML feature selection" step. If in "True" mode label encoding is applied to categorical features.
    
    
* [VIF Feature Selection (Variance Inflation Factor)](https://www.investopedia.com/terms/v/variance-inflation-factor.asp) - Variance inflation factor aims at measuring the amount of multicollinearity in a set of multiple regression variables or features, therefore for this filtering function to be applied all input variables need to be of numeric type. It can be customized by changing the column selection treshold (VIF:float) designated with a default value of 10.
    
    
```py    
    
selected_columns, selected_h2o_importance = atl.feature_selection_h2o(Dataset, # Dataset:pd.DataFrame ,target:str="Name_Target_Column",
                                                                      target,      #  total_vi:float [0.5,1], h2o_fs_models:int [1,50], encoding_fs:bool=True/False
                                                                      total_vi=0.98,     
                                                                      h2o_fs_models =7,
                                                                      encoding_fs=True)


selected_columns, vif_dataset = atl.feature_selection_VIF(Dataset, # Dataset:pd.DataFrame, target:str="Name_Target_Column",
                                                          target,  # VIF:float [3,30]
                                                          VIF=10.0)
```
    
### 2.3 Datetime Feature Engineering

The `eng_date` function converts and transforms columns of Datetime type into additional columns (Day of the week, Weekend, Day of the month, Day of the  Year, Month, Year, Season) which will be added by association to the input dataset and subsequently deletes the original column if variable Drop=True.
    
    
```py   
    
dataset = atl.engin_date(Dataset,Drop=False) # Dataset:pd.DataFrame, Drop:bool
    
```

### 2.4 Predictive Performance Metrics

You can analyse the obtained predictive performance results by using the given bellow functions witch contains the most used metrics for each supervised predictive context.
    
    
```py  

reg_performance = pd.DataFrame(atl.metrics_regression(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
    
binary_class_Performance = pd.DataFrame(atl.metrics_binary_classification(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
    
multiClass_performance = pd.DataFrame(atl.metrics_classification(y_true,y_pred),index=[0])    # y_true:list, y_pred:list
    
```

### 2.5 Extra Auxiliar Functions
    
The following functions were used in the development of this project.
    
```py  
    
## Data Preprocessing 
    
atl.reset_index_DF(Dataset:pd.DataFrame) # return dataset
    
atl.split_dataset(Dataset:pd.DataFrame,
                  Split_Racio:float) # return train, test
    
atl.reindex_columns(Dataset:pd.DataFrame,
                    Feature_Importance:list) # return dataset
        
atl.numerical_columns(Dataset:pd.DataFrame,
                      target:str) # return list_num_cols
    
atl.categorical_columns(Dataset:pd.DataFrame,
                        target:str) # return list_cat_cols 
    
atl.del_nulls_target(Dataset:pd.DataFrame,
                     target:str)  # return Dataset
    
atl.remove_columns_by_nulls(Dataset:pd.DataFrame,
                            percentage:int) # return dataset

## Simplified Null Imputation (Only numeric features)

atl.const_null_imputation(train:pd.DataFrame,
                          test:pd.DataFrame,
                          target:str,
                          imp_value:int=0) # return train, test 
    
atl.simple_null_imputation(train:pd.DataFrame,
                           test:pd.DataFrame,
                           target:str,
                           strat:str='mean') # return train, test
    
atl.knn_null_imputation(train:pd.DataFrame,
                        test:pd.DataFrame,
                        target:str,
                        neighbors:int=5) # return train, test
    
    
atl.iterative_null_imputation(train:pd.DataFrame,
                              test:pd.DataFrame,
                              target:str,
                              order:str='ascending',
                              iterations:int=10) # return train, test
    
    
## VIF Feature Selection Evaluation 
    
atl.vif_performance_selection(train:pd.DataFrame,
                              test:pd.DataFrame,
                              target:str,
                              vif_ratio:float=10.0) # return train, test
    
```   
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/Atlantic/blob/main/LICENSE) for more information.

## Contact 
 
[Luís Santos - LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)   
    

    
