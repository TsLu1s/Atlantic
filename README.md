<br>
<p align="center">
  <h2 align="center"> Atlantic - Automated Data Preprocessing Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `Atlantic` project constitutes an comprehensive and objective approach to simplify and automate data processing through the integration and objectively validated application of various preprocessing mechanisms, ranging from feature engineering, automated feature selection, multiple encoding versions and null imputation methods. The optimization methodology of this framework follows a evaluation structured in tree based models ensembles.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed preprocessing procedures are applicable on any data table associated with Supervised Machine Learning scopes regardless of the properties or specifications of the Dataset features.

* Automated treatment of tabular data associated with predictive analysis: It implements a global and carefully validated tested data treatment based on the characteristics of each Dataset input columns, assuming its identification with the associated target column.

* Robustness and improvement of predictive results: The implementation of the `atlantic` automated data preprocessing pipeline aims at improving predictive performance directly associated with the processing methods implemented based on the Dataset properties.  
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 
   
* [H2O.ai](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* [Optuna](https://optuna.org/)
* [Pandas](https://pandas.pydata.org/)

    
## Framework Architecture <a name = "ta"></a>

<p align="center">
  <img src="https://i.ibb.co/C9dWJmk/ATL-Architecture-Final.png" align="center" width="700" height="680" />
</p>    

## Where to get it <a name = "ta"></a>

Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/atlantic/).  


## Installation  

To install this package from Pypi repository run the following command:

```
pip install atlantic
```

# Usage Examples
    
## 1. Atlantic - Automated Data Preprocessing Pipeline

In order to be able to apply the automated preprocessing `atlantic` pipeline you need first to import the package. 
The following needed step is to load a dataset and define your to be predicted target column name into the variable `target` and define split ratio for your Train and Validation subsets.
You can customize the `fit_processing` method by altering the following running pipeline parameters:
* split_ratio: Division ratio in which the preprocessing methods will be evaluated within the loaded Dataset.
* relevance: Minimal value of the total sum of relative variable\feature importance percentage selected in the `H2O AutoML feature selection` step.
* h2o_fs_models: Quantity of models generated for competition in step `H2O AutoML feature selection` to evaluate the relative importance of each feature (only leaderboard model is selected for evaluation).
* encoding_fs: You can choose if you want to encond your features in order to reduce loading time in `H2O AutoML feature selection` step. If in `True` mode label encoding is applied to categorical features.
* vif_ratio: This value defines the minimal `threshold` for Variance Inflation Factor filtering (default value=10).
 
Importante Notes:
    
* Default predictive evaluation metric for regression contexts is `Mean Absolute Error` and classification is `Accuracy`.
* Although functional, `Atlantic` data processing is not optimized for big data purposes yet.
* Major update is now available in **versions>=1.1.0**
    
```py
    
from atlantic.pipeline import ATLpipeline
import pandas as pd
from sklearn.model_selection import train_test_split 
    
data = pd.read_csv('csv_directory_path', encoding='latin', delimiter=',') # Dataframe Loading Example

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # Required 

### Fit Data Processing

atl = ATLpipeline(X=train,              # X:pd.DataFrame, target:str="Target_Column"
                  target="Target Column")    

atl.fit_processing(split_ratio=0.75,   # split_ratio:float=0.75, relevance:float=0.99 [0.5,1]
                   relevance=0.99,     # h2o_fs_models:int [1,50], encoding_fs:bool=True\False
                   h2o_fs_models=7,    # vif_ratio:float=10.0 [3,30]
                   encoding_fs=True,
                   vif_ratio=10.0)

### Transform Data Processing

train = atl.data_processing(X=train)
test = atl.data_processing(X=test)

### Export Atlantic Preprocessing Metadata

import pickle 
output = open("fit_atl.pkl", 'wb')
pickle.dump(atl, output)
    
```  

## 2. Atlantic - Preprocessing Data
    
### 2.1 Encoding Versions
 
There are multiple preprocessing methods available to direct use. This package provides upgrated encoding `LabelEncoder`, `OneHotEncoder` and [IDF](https://pypi.org/project/cane/) methods with an automatic multicolumn application. 
 
```py
from atlantic.processing import AutoLabelEncoder, AutoIdfEncoder, AutoOneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split 

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # Required

target = "Target_Column" # -> target feature name
    
cat_cols=[col for col in data.select_dtypes(include=['object']).columns if col != target]

### Encoders
## Create Label Encoder
encoder = AutoLabelEncoder()
## Create IDF Encoder
encoder = AutoIdfEncoder()
## Create One-hot Encoder
encoder = AutoOneHotEncoder()

## Fit
encoder.fit(train[cat_cols])

# Transform the DataFrame using Label\IDF\One-hot Encoding
train = encoder.transform(X=train)
test = encoder.transform(X=test)

# Label Encoding : Perform an inverse transform to convert it back the categorical columns values
test = encoder.inverse_transform(X=test)

# IDF & One-hot Encoding : Perform an inverse transform to convert it back the categorical columns values
# Note: Only decodes the last transformed Dataframe
test = encoder.inverse_transform()
            
```    
   
### 2.2 Feature Selection Methods

You can get filter your most valuable features from the dataset via this 2 feature selection methods:
    
* [H2O AutoML Feature Selection](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/variable-importance.html) - This method is based of variable importance evaluation and calculation for tree-based models in H2Os AutoML and it can be customized by use of the following parameters: 
  * relevance: Minimal value of the total sum of relative variable\feature importance percentage selected.
  * h2o_fs_models: Quantity of models generated for competition to evaluate the relative importance of each feature (only leaderboard model will be selected for evaluation).
  * encoding_fs: You can choose if you want to encond your features in order to reduce loading time. If in `True` mode label encoding is applied to categorical features.
    
    
* [VIF Feature Selection (Variance Inflation Factor)](https://www.investopedia.com/terms/v/variance-inflation-factor.asp) - Variance inflation factor aims at measuring the amount of multicollinearity in a set of multiple regression variables or features, therefore for this filtering method to be applied all input variables need to be of numeric type. It can be customized by changing the column filtering treshold `vif_threshold` designated with a default value of 10.
    
    
```py    
from atlantic.selector import Selector

fs=Selector(X=train,target="Target_Column")

cols_vif = fs.feature_selection_vif(vif_threshold=10.0)   # X: Only numerical values allowed & No nans allowed in VIF

selected_cols, selected_importance = fs.feature_selection_h2o(relevance=0.99,     # relevance:float [0.5,1], h2o_fs_models:int [1,50]
                                                              h2o_fs_models=7,    # encoding_fs:bool=True/False
                                                              encoding_fs=True)
```
    
### 2.3 Null Imputation Auxiliar Methods
    
Simplified and automated multivariate null imputation methods based from [Sklearn](https://scikit-learn.org/stable/modules/impute.html) are also provided and applicable, as following:
    
```py  

## Simplified Null Imputation (Only numeric features)
from atlantic.imputation import AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer

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

```   

## Citation

Feel free to cite Atlantic as following:

```

@article{SANTOS2023100532,
  author = {Luis Santos and Luis Ferreira}
  title = {Atlantic - Automated data preprocessing framework for supervised machine learning},
  journal = {Software Impacts},
  volume = {17},
  year = {2023},
  issn = {2665-9638},
  doi = {http://dx.doi.org/10.1016/j.simpa.2023.100532},
  url = {https://www.sciencedirect.com/science/article/pii/S2665963823000696}
}

```

    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/Atlantic/blob/main/LICENSE) for more information.

## Contact 
 
[Luis Santos - LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
