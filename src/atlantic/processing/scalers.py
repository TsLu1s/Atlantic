import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import TransformerMixin

  
class AutoMinMaxScaler(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoMinMaxScaler class. This class provides functionality for automatically scaling
        numerical features using Min-Max Scaling, which scales the data to a fixed range, typically [0, 1].
        
        Attributes:
        scaler (MinMaxScaler): An instance of MinMaxScaler to perform the scaling.
        columns (Index): Stores the column names of the DataFrame being transformed.
        """
        self.scaler= MinMaxScaler()
        self.columns = None
        
    def fit(self, X, y=None):
        """
        Fit the MinMaxScaler to the data. This method computes the minimum and maximum values of each feature to be
        used for scaling.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be scaled.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoMinMaxScaler.
        """
        self.columns = X.columns
        self.scaler.fit(X)
        
        return self
        
    def transform(self, X, y=None):
        """
        Transform the data using the fitted MinMaxScaler. This method scales the numerical features to the range [0, 1].
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be scaled.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the scaled numerical features.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled,columns=self.columns)
    
    def inverse_transform(self,X):
        """
        Inverse transform the scaled data back to the original scale. This method reverses the scaling operation.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the scaled numerical features.
        
        Returns:
        pd.DataFrame: A DataFrame with the numerical features in the original scale.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(X_original, columns=self.columns)

class AutoStandardScaler(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoStandardScaler class. This class provides functionality for automatically
        scaling numerical features using Standard Scaling, which scales the data to have a mean of 0 and a standard deviation
        of 1.
        
        Attributes:
        scaler (StandardScaler): An instance of StandardScaler to perform the scaling.
        columns (Index): Stores the column names of the DataFrame being transformed.
        """
        self.scaler = StandardScaler()
        self.columns=None
        
    def fit(self, X, y=None):
        """
        Fit the StandardScaler to the data. This method computes the mean and standard deviation of each feature to be
        used for scaling.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be scaled.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoStandardScaler.
        """
        self.columns=X.columns
        self.scaler.fit(X)
        
        return self 
    
    def transform(self, X, y=None):
        """
        Transform the data using the fitted StandardScaler. This method scales the numerical features to have a mean of 0
        and a standard deviation of 1.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be scaled.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the scaled numerical features.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.columns)
    
    def inverse_transform(self, X):
        """
        Inverse transform the scaled data back to the original scale. This method reverses the scaling operation.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the scaled numerical features.
        
        Returns:
        pd.DataFrame: A DataFrame with the numerical features in the original scale.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(X_original, columns=self.columns)
    
class AutoRobustScaler(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoRobustScaler class. This class provides functionality for automatically
        scaling numerical features using Robust Scaling, which scales the data based on robust statistics to handle
        outliers.
        
        Attributes:
        scaler (RobustScaler): An instance of RobustScaler to perform the scaling.
        columns (Index): Stores the column names of the DataFrame being transformed.
        """
        self.scaler = RobustScaler()
        self.columns = None
        
    def fit(self, X, y=None):
        """
        Fit the RobustScaler to the data. This method computes the robust statistics of each feature to be used for scaling.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be scaled.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoRobustScaler.
        """
        self.columns = X.columns
        self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the data using the fitted RobustScaler. This method scales the numerical features based on robust
        statistics to handle outliers.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be scaled.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the scaled numerical features.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns)
    
    def inverse_transform(self, X):
        """
        Inverse transform the scaled data back to the original scale. This method reverses the scaling operation.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the scaled numerical features.
        
        Returns:
        pd.DataFrame: A DataFrame with the numerical features in the original scale.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(X_original, columns=self.columns)
