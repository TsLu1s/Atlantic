import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Analysis:
    def __init__(self, target: str = None):
        # Constructor for the Analysis class, initializes the target attribute.
        self.target = target
        self._label_encoder=None

    def split_dataset(self, X: pd.DataFrame, split_ratio: float = 0.75):
        # Splits the dataset into train and test sets based on the split_ratio.
        assert 0.5 <= split_ratio < 0.95, 'split_ratio should be in [0.5, 0.95) interval'
        X = X.dropna(subset=[self.target])
        
        return train_test_split(X, train_size=split_ratio)

    def divide_dfs(self, train: pd.DataFrame, test: pd.DataFrame):
        # Divides the DataFrames into feature and target sets.
        X_train, X_test = train.drop(self.target, axis=1), test.drop(self.target, axis=1)
        y_train, y_test = train[self.target], test[self.target]
        
        if self.target_type(X=train)[0]=="Class":
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            y_test = self._label_encoder.transform(y_test)
        
        return X_train, X_test, y_train, y_test

    def target_type(self, X: pd.DataFrame):
        # Determines the prediction type and evaluation metric based on target data type.
        target_dtype = X[self.target].dtype
        pred_type, eval_metric = 'Reg', 'Mean Absolute Error'
        if target_dtype not in ['int32', 'int64', 'float32', 'float64']:
            pred_type, eval_metric = 'Class', 'Accuracy'
            
        return pred_type, eval_metric

    def num_cols(self, X: pd.DataFrame):
        # Returns a list of numerical columns (int64 and float64) in the DataFrame.
        return [col for col in X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns if col != self.target]

    def cat_cols(self, X: pd.DataFrame):
        # Returns a list of categorical columns (object data type) in the DataFrame.
        return [col for col in X.select_dtypes(include=['object']).columns if col != self.target]

    @staticmethod
    def remove_columns_by_nulls(X: pd.DataFrame, percentage: float):
        # Removes columns with null values exceeding a specified percentage threshold.
        total_rows = len(X)
        min_count = int((1 - percentage / 100) * total_rows)
        X_ = X.dropna(axis=1, thresh=min_count)
        
        return X_
    
    @staticmethod
    def engin_date(X: pd.DataFrame, drop: bool = True):
        # This method is responsible for engineering date-related features in the input DataFrame X.
        # It can also optionally drop the original datetime columns based on the 'drop' parameter.
    
        # Extract the data types of each column in X and create a DataFrame x.
        x = pd.DataFrame(X.dtypes)
        # Create a 'column' column to store the column names.
        x['column'] = x.index
        # Reset the index and drop the original index column.
        x = x.reset_index().drop(['index'], axis=1).rename(columns={0: 'dtype'})
        # Filter for columns with datetime data type.
        a = x.loc[x['dtype'] == 'datetime64[ns]']
    
        # Initialize an empty list to store the names of datetime columns.
        list_date_columns = []
    
        # Loop through datetime columns and perform feature engineering.
        for date_col in a['column']:
            list_date_columns.append(date_col)
            # Convert datetime values to a standardized format (Year-Month-Day Hour:Minute:Second).
            X[date_col] = pd.to_datetime(X[date_col].dt.strftime('%Y-%m-%d %H:%M:%S'))
    
        # Define a function to create additional date-related features for a given column.
        def create_date_features(df, elemento):
            # Extract day of the month, day of the week, and whether it's a weekend.
            df[elemento + '_day_of_month'] = df[elemento].dt.day
            df[elemento + '_day_of_week'] = df[elemento].dt.dayofweek + 1
            df[[elemento + '_is_wknd']] = df[[elemento + '_day_of_week']].replace([1, 2, 3, 4, 5, 6, 7],
                                                                                  [0, 0, 0, 0, 0, 1, 1])
            df[elemento + '_month'] = df[elemento].dt.month
            df[elemento + '_day_of_year'] = df[elemento].dt.dayofyear
            df[elemento + '_year'] = df[elemento].dt.year
            df[elemento + '_hour'] = df[elemento].dt.hour
            df[elemento + '_minute'] = df[elemento].dt.minute
            df[elemento + '_second'] = df[elemento].dt.second
    
            return df
    
        # Loop through the list of date columns and create the additional features using the defined function.
        for elemento in list_date_columns:
            X = create_date_features(X, elemento)
            # If 'drop' is set to True, drop the original datetime column.
            if drop == True: X = X.drop(elemento, axis=1)
    
        # Return the DataFrame X with the engineered date-related features.
        return X




