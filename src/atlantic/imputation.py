import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.base import TransformerMixin

class AutoSimpleImputer(TransformerMixin):
    def __init__(self, 
                 strategy:str='mean', # {"mean", "median", "most_frequent", "constant"}
                 target:str=None):
        self.strategy = strategy 
        self.target = target
        self.imputer = None
        self.numeric_columns = None  # Store the numeric columns here

    def fit(self, X, y=None):
        if self.target is not None:
            X = X.drop(columns=[self.target])
        
        # Detect numeric columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit the imputer on numeric columns only
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X[self.numeric_columns])
        
        return self

    def transform(self, X):
        if self.imputer is None:
            raise ValueError("You must call 'fit' first to initialize the imputer.")
        
        # Transform only the numeric columns using the fitted imputer
        imputed_numeric = self.imputer.transform(X[self.numeric_columns])
        
        # Update the original DataFrame with imputed values
        X[self.numeric_columns] = imputed_numeric
        
        return X
    
    def impute(self,train:pd.DataFrame, test:pd.DataFrame, strategy:str="mean"):
        if strategy is not None:
            self.strategy = strategy  # Update the strategy if provided
        
        # Fit the AutoSimpleImputer instance on the training data
        self.fit(train)
        
        # Transform both the train and test data using the fitted imputer
        train = self.transform(train.copy())
        test = self.transform(test.copy())
        
        return train, test

class AutoKNNImputer(TransformerMixin):
    def __init__(self, n_neighbors:int=5,
                 weights:str="uniform", #{"uniform", "distance"}
                 target:str=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.target = target
        self.imputer = None
        self.numeric_columns = None  # Store the numeric columns here

    def fit(self, X, y=None):
        if self.target is not None:
            X = X.drop(columns=[self.target])
        
        # Detect numeric columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit the imputer on numeric columns only
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors,weights=self.weights)
        self.imputer.fit(X[self.numeric_columns])
        
        return self

    def transform(self, X):
        if self.imputer is None:
            raise ValueError("You must call 'fit' first to initialize the imputer.")
        
        # Transform only the numeric columns using the fitted imputer
        imputed_numeric = self.imputer.transform(X[self.numeric_columns])
        
        # Update the original DataFrame with imputed values
        X[self.numeric_columns] = imputed_numeric
        
        return X

class AutoIterativeImputer(TransformerMixin):

    def __init__(self, max_iter:int=10, 
                 random_state:int=None, 
                 initial_strategy:str="mean", # {"mean", "median", "most_frequent", "constant"}
                 imputation_order:str="ascending", # {"ascending", "descending", "roman", "arabic", "random"}
                 target:str=None):
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.target = target
        self.imputer = None
        self.numeric_columns = None  # Store the numeric columns here

    def fit(self, X, y=None):
        if self.target is not None:
            X = X.drop(columns=[self.target])
        
        # Detect numeric columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit the imputer on numeric columns only
        self.imputer = IterativeImputer(max_iter=self.max_iter,
                                        random_state=self.random_state,
                                        initial_strategy=self.initial_strategy,
                                        imputation_order=self.imputation_order)
        self.imputer.fit(X[self.numeric_columns])
        
        return self

    def transform(self, X):
        if self.imputer is None:
            raise ValueError("You must call 'fit' first to initialize the imputer.")
        
        # Transform only the numeric columns using the fitted imputer
        imputed_numeric = self.imputer.transform(X[self.numeric_columns])
        
        # Update the original DataFrame with imputed values
        X[self.numeric_columns] = imputed_numeric
        
        return X
  
    


