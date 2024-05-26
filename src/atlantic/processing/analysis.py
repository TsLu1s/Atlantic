import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Analysis:
    def __init__(self, target: str = None):
        """
        Initialize an instance of the Analysis class. The Analysis class is designed to facilitate the preprocessing
        and preparation of datasets for machine learning tasks. It includes methods for splitting datasets,
        encoding target variables, and engineering features.
        
        Parameters:
        target (str): The name of the target variable column. This is the dependent variable that the model will learn to predict.
        
        Attributes:
        target (str): Stores the name of the target variable provided during initialization. This attribute is used by other methods
                      in the class to identify and manipulate the target variable.
        _label_encoder (LabelEncoder): A private attribute that is used to encode categorical target variables. It is initialized as None
                                       and is instantiated when needed.
        n_dtypes (list): A list of numerical data types. This attribute helps in identifying numerical columns in the dataset.
                         The list includes common integer and floating-point data types used in pandas DataFrames.
        """
        self.target = target
        self._label_encoder = None
        self.n_dtypes = ['int','int32', 'int64','float','float32', 'float64']

    def split_dataset(self, X: pd.DataFrame, split_ratio: float = 0.75):
        """
        Splits the dataset into training and testing sets based on the specified split ratio. The method ensures that the split ratio
        is within a sensible range and handles any missing values in the target column by dropping those rows.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the features and the target variable. It is expected that the DataFrame
                          includes a column with the name matching the target attribute.
        split_ratio (float): The proportion of the dataset to be used for the training set. The value must be between 0.5 and 0.98
                             to ensure that both training and testing sets are sufficiently large for meaningful analysis.
        """
        # Splits the dataset into train and test sets based on the split_ratio.
        assert 0.5 <= split_ratio <= 0.98, 'split_ratio should be in [0.5, 0.98] interval'
        X = X.dropna(subset=[self.target])
        
        return train_test_split(X, train_size = split_ratio)

    def divide_dfs(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        Divides the training and testing DataFrames into feature sets (X) and target sets (y). It separates the target variable from the
        features and encodes the target variable if it is categorical.
        
        Parameters:
        train (pd.DataFrame): The DataFrame containing the training data. It includes both features and the target variable.
        test (pd.DataFrame): The DataFrame containing the testing data. It also includes both features and the target variable.
        
        Returns:
        tuple: A tuple containing four elements:
               - X_train (pd.DataFrame): The feature set for the training data.
               - X_test (pd.DataFrame): The feature set for the testing data.
               - y_train (pd.Series or np.ndarray): The target set for the training data.
               - y_test (pd.Series or np.ndarray): The target set for the testing data.

        """
        X_train, X_test = train.drop(self.target, axis=1), test.drop(self.target, axis=1)
        y_train, y_test = train[self.target], test[self.target]
        
        if self.target_type(X = train)[0] == "Class":
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            y_test = self._label_encoder.transform(y_test)
        
        return X_train, X_test, y_train, y_test

    def target_type(self, X: pd.DataFrame):
        """
        Determines the type of the target variable and the appropriate evaluation metric based on the target's data type.
        If the target variable is numerical, it is considered a regression problem. If it is categorical, it is considered a
        classification problem.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the target variable.
        
        Returns:
        tuple: A tuple containing two elements:
               - pred_type (str): The type of prediction task ('Reg' for regression, 'Class' for classification).
               - eval_metric (str): The default evaluation metric for the prediction task ('Mean Absolute Error' for regression,
                                    'Accuracy' for classification).
        """
        target_dtype = X[self.target].dtype
        pred_type, eval_metric = 'Reg', 'Mean Absolute Error'
        if target_dtype not in self.n_dtypes:
            pred_type, eval_metric = 'Class', 'Accuracy'
            
        return pred_type, eval_metric

    def num_cols(self, X: pd.DataFrame):
        """
        Identifies and returns a list of numerical columns in the input DataFrame, excluding the target variable.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the dataset.
        
        Returns:
        list: A list of column names corresponding to numerical columns in the DataFrame, excluding the target variable.
        """
        return [col for col in X.select_dtypes(include=self.n_dtypes).columns if col != self.target]

    def cat_cols(self, X: pd.DataFrame):
        """
        Identifies and returns a list of categorical columns in the input DataFrame, excluding the target variable.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the dataset.
        
        Returns:
        list: A list of column names corresponding to categorical columns in the DataFrame, excluding the target variable.
        """
        return [col for col in X.select_dtypes(include=['object','category']).columns if col != self.target]

    @staticmethod
    def remove_columns_by_nulls(X: pd.DataFrame, percentage: float):
        """
        Removes columns from the DataFrame that have a percentage of null values greater than the specified threshold.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame from which columns with excessive null values will be removed.
        percentage (float): The threshold percentage of null values. Columns with a higher percentage of null values will be removed.
        
        Returns:
        pd.DataFrame: The DataFrame with columns having excessive null values removed.
        """
        min_count = int((1 - percentage / 100) * X.shape[0])
        X_ = X.dropna(axis = 1, thresh = min_count)
        
        return X_
    
    @staticmethod
    def engin_date(X: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """
        Engineer date-related features in the input DataFrame. This method extracts various temporal features from
        datetime columns and optionally drops the original datetime columns.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing datetime columns.
        drop (bool): If True, the original datetime columns are dropped after feature engineering. Default is True.
        
        Returns:
        pd.DataFrame: A DataFrame with the original datetime columns transformed into multiple date-related features.
        """
        
        # Identify datetime columns in the DataFrame.
        datetime_columns = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        for col in datetime_columns:
            # Convert datetime values to a standardized format (Year-Month-Day Hour:Minute:Second).
            X[col] = pd.to_datetime(X[col].dt.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Create additional date-related features.
            X[col + '_day_of_month'] = X[col].dt.day
            X[col + '_day_of_week'] = X[col].dt.dayofweek + 1
            X[col + '_is_wknd'] = X[col + '_day_of_week'].isin([6, 7]).astype(int)
            X[col + '_month'] = X[col].dt.month
            X[col + '_day_of_year'] = X[col].dt.dayofyear
            X[col + '_year'] = X[col].dt.year
            X[col + '_hour'] = X[col].dt.hour
            X[col + '_minute'] = X[col].dt.minute
            X[col + '_second'] = X[col].dt.second
            
            # Drop the original datetime column if 'drop' is set to True.
            if drop:
                X = X.drop(columns=col)
        
        return X
    
    
