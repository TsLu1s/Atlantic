import pandas as pd
from atlantic.processing import AutoMinMaxScaler, AutoStandardScaler, AutoLabelEncoder, AutoIdfEncoder, Encoding_Version
from atlantic.imputation import AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer
from atlantic.analysis import Analysis
from atlantic.selector import Selector
from atlantic.pattern import Pattern

class ATLpipeline(Selector):
    def __init__(self,
                 X:pd.DataFrame=None,
                 target:str=None):
        super().__init__(X, target)
        self.enc_method=None
        self.imp_method=None
        self.encoder=None
        self.scaler=None
        self.imputer=None
        self.n_cols=None
        self.c_cols=None
        self.cols=None
        self.h2o_feature_importance=None

    def fit_processing(self,
                       split_ratio:float,
                       relevance:float=0.99,
                       h2o_fs_models:int=7,
                       encoding_fs:bool=True,
                       vif_ratio:float=10.0):
    
    ### Data Treatment
        if self.pred_type=='Class': self.X[self.target]=self.X[self.target].astype(str)
            
        X_ = self.X.copy()

        X_ = super().remove_columns_by_nulls(X=X_,percentage=99.9)
        
        X_ = X_[[col for col in X_.columns if col != self.target] + [self.target]] # target to last index
        
        X_ = super().engin_date(X=X_, drop=True)
        dataset=X_.copy()
        
        train, test = super().split_dataset(X=X_,split_ratio=split_ratio)
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
        
    ### Feature Selection 
        fs=Selector(X=train,target=self.target)
        sel_cols, self.h2o_feature_importance=fs.feature_selection_h2o(relevance=relevance,
                                                                       h2o_fs_models=h2o_fs_models,
                                                                       encoding_fs=encoding_fs)
        train, test = train[sel_cols], test[sel_cols]
    #### Encoding Method Selection    
        pat=Pattern(train=train,test=test,target=self.target)          
        self.enc_method,perf_ = pat.encoding_selection(),pat.perf
        
    ### Null Imputation Selection 
        if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
            train, test = pat.imputation_selection()
            self.imp_method,perf_=pat.imp_method,pat.perf
        else:
            self.imp_method='Undefined'
            ev = Encoding_Version(train=train, test=test, target=self.target)
            if self.enc_method=='Encoding Version 1':
                train, test=ev.encoding_v1()
            elif self.enc_method=='Encoding Version 2':
                train, test=ev.encoding_v2()
            elif self.enc_method=='Encoding Version 3':
                train, test=ev.encoding_v3()
            elif self.enc_method=='Encoding Version 4':
                train, test=ev.encoding_v4()
            print('There are no missing values in the Dataset')  

    ### Variance Inflation Factor (VIF) Application 
        train, test = Pattern(train=train,test=test,target=self.target).vif_performance(vif_threshold=vif_ratio,perf_=perf_)
        self.cols=list(train.columns)
    
    ### Fit Processors 
        dataset=dataset[self.cols]
        self.n_cols=list(super().num_cols(X=dataset))
        self.c_cols=list(super().cat_cols(X=dataset))
        
        ## Fit Encoding Version
        if len(self.n_cols)>0:
            if self.enc_method=='Encoding Version 1' or self.enc_method=='Encoding Version 3':
                    self.scaler = AutoStandardScaler()
                    self.scaler.fit(X=dataset[self.n_cols])
                    dataset[self.n_cols]=self.scaler.transform(X=dataset[self.n_cols])
            if self.enc_method=='Encoding Version 2' or self.enc_method=='Encoding Version 4':
                    self.scaler = AutoMinMaxScaler()
                    self.scaler.fit(X=dataset[self.n_cols])
                    dataset[self.n_cols]=self.scaler.transform(X=dataset[self.n_cols])
        if len(self.c_cols)>0:
            if self.enc_method=='Encoding Version 1' or self.enc_method=='Encoding Version 2':
                    self.encoder = AutoIdfEncoder()
                    self.encoder.fit(dataset[self.c_cols])
                    dataset=self.encoder.transform(X=dataset)
            if self.enc_method=='Encoding Version 3' or self.enc_method=='Encoding Version 4':
                    self.encoder = AutoLabelEncoder()
                    self.encoder.fit(dataset[self.c_cols])
                    dataset=self.encoder.transform(X=dataset)
            
        ## Fit Null Imputation
        if self.imp_method=='Simple':
            self.imputer = AutoSimpleImputer(strategy='mean')
            self.imputer.fit(dataset)  # Fit on DataFrame
        
        elif self.imp_method=='KNN':
            self.imputer = AutoKNNImputer(n_neighbors=3,
                                          weights="uniform")
            self.imputer.fit(dataset)  # Fit on DataFrame
        
        elif self.imp_method=='Iterative':
            self.imputer = AutoIterativeImputer(max_iter=10,
                                                random_state=42,
                                                initial_strategy="mean",
                                                imputation_order="ascending")
            self.imputer.fit(dataset)  # Fit on DataFrame
        
        return self
        
    def data_processing(self,X:pd.DataFrame):
        
    ### Transformation Proceedment
        X_=X.copy()
        X_=Analysis.engin_date(X_)
        X_=X_[self.cols]
        
        if len(self.n_cols)>0:
            X_[self.n_cols] = self.scaler.transform(X_[self.n_cols])
        if len(self.c_cols)>0:
            X_=self.encoder.transform(X=X_)
        if self.imp_method != "Undefined" and X_[self.c_cols+self.n_cols].isnull().sum().sum() != 0:
            X_=self.imputer.transform(X_.copy())
    
        return X_
    
    
    
    