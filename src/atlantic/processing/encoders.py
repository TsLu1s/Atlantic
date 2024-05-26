import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin
import cane
       
class AutoLabelEncoder(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoLabelEncoder class. This class provides functionality for automatically
        encoding categorical features using Label Encoding, where each unique category is converted to a numerical value.
        
        Attributes:
        label_encoders (dict): A dictionary to store the LabelEncoder instance for each column.
        columns (Index): Stores the column names of the DataFrame being transformed.
        """
        self.label_encoders={}
        self.columns=None
    
    def fit(self, X, y=None):
        """
        Fit the LabelEncoders to each column in the DataFrame. This method prepares the encoder for transforming the
        categorical data into numerical data.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be encoded.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoLabelEncoder.
        """
        self.columns=X.columns
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            #Store the original classes and add a new label for unseen values
            le.classes_ = np.append(le.classes_, "Unknown")
            self.label_encoders[col] = le
            
        return self
    
    def transform(self, X, y=None):
        """
        Transform the categorical features in the DataFrame into numerical values using the fitted LabelEncoders. This
        method also handles unseen values by encoding them as "Unknown".
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be transformed.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the categorical features encoded as numerical values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_encoded = X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            # Use the "UNKNOWN" label for previously unseen values
            X_encoded[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["Unknown"])[0])

        return X_encoded
    
    def inverse_transform(self, X):
        """
        Inverse transform the numerical values back to the original categorical values using the fitted LabelEncoders.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the numerical features to be inverse transformed.
        
        Returns:
        pd.DataFrame: A DataFrame with the numerical features decoded back to their original categorical values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_decoded=X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            X_decoded[col]=le.inverse_transform(X[col])
            
        return X_decoded
        
class AutoOneHotEncoder(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoOneHotEncoder class. This class provides functionality for automatically
        encoding categorical features using One-Hot Encoding, where each unique category is converted to a binary vector.
        
        Attributes:
        one_hot_encoders (dict): A dictionary to store the OneHotEncoder instance for each column.
        columns (Index): Stores the column names of the DataFrame being transformed.
        decoded (pd.DataFrame): A copy of the original DataFrame before transformation, used for inverse transformation.
        """
        self.one_hot_encoders = {}
        self.columns = None
        self.decoded = None

    def fit(self, X, y=None):
        """
        Fit the OneHotEncoders to each column in the DataFrame. This method prepares the encoder for transforming the
        categorical data into binary vectors.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be encoded.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        self: The fitted instance of AutoOneHotEncoder.
        """
        self.columns = X.columns
        
        for col in X.columns:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoder.fit(X[col].values.reshape(-1, 1))
            self.one_hot_encoders[col] = encoder
            
        return self

    def transform(self, X, y=None):
        """
        Transform the categorical features in the DataFrame into binary vectors using the fitted OneHotEncoders. This
        method also handles unseen values by adding columns of zeros for unknown categories.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be transformed.
        y: Ignored. This parameter is included for compatibility with the TransformerMixin interface.
        
        Returns:
        pd.DataFrame: A DataFrame with the categorical features encoded as binary vectors.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")

        X_encoded = X.copy()
        self.decoded = X.copy()

        for col in self.columns:
            if col in X_encoded.columns:
                encoder = self.one_hot_encoders[col]
                encoded_col = encoder.transform(X_encoded[col].values.reshape(-1, 1))

                # Handle unknown categories by adding a column of zeros
                if len(encoder.get_feature_names_out([col])) > 1:
                    unknown_cols = [col for col in encoder.get_feature_names_out([col]) if
                                    col.startswith(f"{col}_Unknown")]
                    for unknown_col in unknown_cols:
                        X_encoded[unknown_col] = 0

                encoded_columns = encoder.get_feature_names_out([col])
                X_encoded[encoded_columns] = encoded_col

                # Drop the original column
                X_encoded.drop(columns=[col], inplace=True)

        return X_encoded
    
    def inverse_transform(self):
        """
        Inverse transform the binary vectors back to the original categorical values using the stored original DataFrame.
        
        Returns:
        pd.DataFrame: The original DataFrame before transformation.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
                    
        return self.decoded
        
class AutoIdfEncoder(TransformerMixin):
    def __init__(self):
        """
        Initialize an instance of the AutoIdfEncoder class. This class provides functionality for encoding categorical features
        using an IDF (Inverse Document Frequency) based approach, which is a technique often used in text processing to evaluate
        how important a word is to a document in a collection.
        
        Attributes:
        idf_fit (dict): A dictionary to store the IDF mapping for each column.
        columns (Index): Stores the column names of the DataFrame being transformed.
        """
        self.idf_fit = None
        self.columns = None

    def fit(self, X):
        """
        Fit the IDF encoders to each column in the DataFrame. This method prepares the encoder for transforming the
        categorical data based on the IDF values.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be encoded.
        
        Returns:
        self: The fitted instance of AutoIdfEncoder.
        
        Raises:
        ValueError: If the `columns` attribute is not specified during initialization.
        """
        self.columns = X.columns
        
        if self.columns is None:
            raise ValueError("You must specify categorical_columns when creating the AutoIdfEncoding instance.")
        
        IDF_filter = cane.idf(X, n_coresJob=2, disableLoadBar=True, columns_use=self.columns)
        self.idf_fit = cane.idfDictionary(Original=X, Transformed=IDF_filter, columns_use=self.columns)
        
        return self

    def transform(self, X):
        """
        Transform the categorical features in the DataFrame based on the fitted IDF values. This method maps each category
        to its corresponding IDF value.
        
        Parameters:
        X (pd.DataFrame): The input DataFrame containing the categorical features to be transformed.
        
        Returns:
        pd.DataFrame: A DataFrame with the categorical features encoded based on their IDF values.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.idf_fit is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_encoded = X.copy()
        self.decoded = X.copy()

        for col in self.columns:
            X_encoded[col] = X_encoded[col].map(self.idf_fit[col]).fillna(max(self.idf_fit[col].values()))
            
        return X_encoded
    
    def inverse_transform(self):
        """
        Inverse transform the IDF-encoded values back to the original categorical values using the stored original DataFrame.
        
        Returns:
        pd.DataFrame: The original DataFrame before transformation.
        
        Raises:
        ValueError: If the transformer has not been fitted yet.
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
                    
        return self.decoded