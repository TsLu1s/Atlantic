import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin
import cane
       
class AutoLabelEncoder(TransformerMixin):
    def __init__(self):
        self.label_encoders={}
        self.columns=None
    
    def fit(self, X, y=None):
        self.columns=X.columns
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            #Store the original classes and add a new label for unseen values
            le.classes_ = np.append(le.classes_, "Unknown")
            self.label_encoders[col] = le
            
        return self
    
    def transform(self, X, y=None):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_encoded = X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            # Use the "UNKNOWN" label for previously unseen values
            X_encoded[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["Unknown"])[0])
            
        return X_encoded
    
    def inverse_transform(self, X):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_decoded=X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            X_decoded[col]=le.inverse_transform(X[col])
            
        return X_decoded
        
class AutoOneHotEncoder(TransformerMixin):
    def __init__(self):
        self.one_hot_encoders = {}
        self.columns = None
        self.decoded = None

    def fit(self, X, y=None):
        self.columns = X.columns
        
        for col in X.columns:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoder.fit(X[col].values.reshape(-1, 1))
            self.one_hot_encoders[col] = encoder
            
        return self

    def transform(self, X, y=None):
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
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
                    
        return self.decoded
        
class AutoIdfEncoder(TransformerMixin):
    def __init__(self):
        self.idf_fit = None
        self.columns = None

    def fit(self, X):
        self.columns = X.columns
        
        if self.columns is None:
            raise ValueError("You must specify categorical_columns when creating the AutoIdfEncoding instance.")
        
        IDF_filter = cane.idf(X, n_coresJob=2, disableLoadBar=True, columns_use=self.columns)
        self.idf_fit = cane.idfDictionary(Original=X, Transformed=IDF_filter, columns_use=self.columns)
        
        return self

    def transform(self, X):
        if self.idf_fit is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        X_encoded = X.copy()
        self.decoded = X.copy()

        for col in self.columns:
            X_encoded[col] = X_encoded[col].map(self.idf_fit[col]).fillna(max(self.idf_fit[col].values()))
            
        return X_encoded
    
    def inverse_transform(self):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
                    
        return self.decoded
    
class AutoMinMaxScaler(TransformerMixin):
    def __init__(self):
        self.scaler= MinMaxScaler()
        self.columns = None
        
    def fit(self, X, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        
        return self
        
    def transform(self, X, y=None):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled,columns=self.columns)
    
    def inverse_transform(self,X):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(X_original, columns=self.columns)

class AutoStandardScaler(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns=None
        
    def fit(self, X, y=None):
        self.columns=X.columns
        self.scaler.fit(X)
        
        return self 
    
    def transform(self, X, y=None):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.columns)
    
    def inverse_transform(self, X):
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet.")
            
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(X_original, columns=self.columns)

class Encoding_Version:
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame, target:str):
        self.train = train
        self.test = test
        self.target = target
        self.cat_cols = [col for col in self.train.select_dtypes(include=['object']).columns if col != self.target]
        self.num_cols = [col for col in self.train.select_dtypes(include=['int64', 'float64']).columns if col != self.target]
        
    @staticmethod
    def apply_encoder(train, test, encoder_, cols):
        if len(cols) > 0:
            encoder_instance = encoder_()  # Instantiate the selected encoder
            encoder_instance.fit(X=train[cols])
            train[cols] = encoder_instance.transform(X=train[cols].copy())
            test[cols] = encoder_instance.transform(X=test[cols].copy())

    def encoding_v1(self):
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoStandardScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoIdfEncoder, self.cat_cols)
        return train_enc, test_enc

    def encoding_v2(self):
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoMinMaxScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoIdfEncoder, self.cat_cols)
        return train_enc, test_enc

    def encoding_v3(self):
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoStandardScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoLabelEncoder, self.cat_cols)
        return train_enc, test_enc

    def encoding_v4(self):
        train_enc, test_enc = self.train.copy(), self.test.copy()
        self.apply_encoder(train_enc, test_enc, AutoMinMaxScaler, self.num_cols)
        self.apply_encoder(train_enc, test_enc, AutoLabelEncoder, self.cat_cols)
        return train_enc, test_enc
