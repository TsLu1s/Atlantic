import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from statsmodels.stats.outliers_influence import variance_inflation_factor
from atlantic.processing import AutoLabelEncoder
from atlantic.analysis import Analysis

class Selector(Analysis):
    def __init__(self, 
                 X:pd.DataFrame, 
                 target:str):
        super().__init__(target)
        self.X = X
        self.vif_df = None
        self.pred_type,self.eval_metric=super().target_type(X=X)
        
    def calculate_vif(self,X:pd.DataFrame):
        # Calculate Variance Inflation Factor (VIF) for numeric columns in the dataset.
        # The VIF measures multicollinearity between variables.
        
        # Check if there are any categorical columns or null values in X
        if len([col for col in X[list(X.columns)].select_dtypes(include=['number']).columns if col != self.target]) < len(list(X.columns))-1: 
            raise ValueError("Only numerical columns are supported in VIF calculation.")
        if X.isnull().values.any():
            raise ValueError("Null values are not supported in VIF calculation.")

        vif = pd.DataFrame()
        vif['variables'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif.sort_values(['VIF'], ascending=False)
        
        return vif

    def feature_selection_vif(self,vif_threshold:float=10.0):
        # Perform feature selection using VIF (Variance Inflation Factor).
        assert vif_threshold >= 3 and vif_threshold <= 30, 'VIF threshold should be in [3, 30] interval'
        cols = list(self.X.columns)
        cols.remove(self.target)
        X_ = self.X[cols].copy()
        self.vif_df = self.calculate_vif(X_)

        while self.vif_df['VIF'].max() >= vif_threshold:
            # Iteratively remove columns with VIF above the threshold.
            self.vif_df.drop(self.vif_df['variables'].loc[self.vif_df['VIF'] == self.vif_df['VIF'].max()].index,
                             inplace=True)
            cols = [rows for rows in self.vif_df['variables']]
            X_ = X_[cols]
            self.vif_df = self.calculate_vif(X_)
        cols.append(self.target)
        
        return cols
        
    def feature_selection_h2o(self,relevance:float=0.99,h2o_fs_models:int=7,encoding_fs:bool=True):
        # Perform feature selection using H2O AutoML and relevance percentage.
        assert relevance>=0.5 and relevance<=1 , 'relevance value should be in [0.5,1] interval'
        assert h2o_fs_models>=1 and h2o_fs_models<=50 , 'h2o_fs_models value should be in [0,50] interval'
    
        # Initialize H2O and prepare the data for feature selection.
        h2o.init()
        
        X_=self.X.copy()
        
        if encoding_fs==True:
            encoder = AutoLabelEncoder()
            encoder.fit(X_[[col for col in X_.select_dtypes(include=['object']).columns if col != self.target]])
            X_=encoder.transform(X=X_)
        
        input_cols=list(X_.columns)
        input_cols.remove(self.target)
            
        train_h2o=h2o.H2OFrame(X_)

        if self.pred_type=='Class': train_h2o[self.target] = train_h2o[self.target].asfactor()
        # Train an AutoML model and retrieve the leaderboard.
        aml = H2OAutoML(max_models=h2o_fs_models,
                        nfolds=3,
                        seed=1,
                        exclude_algos=['GLM',  
                                       'DeepLearning', 
                                       'StackedEnsemble'],
                        sort_metric='AUTO')
        aml.train(x=input_cols,y=self.target,training_frame=train_h2o)
        leaderboards = aml.leaderboard
        leaderboards_df = leaderboards.as_data_frame()
        print(leaderboards_df)
        
        list_id_model,sel_cols=[],[]
        for row in leaderboards_df['model_id']: list_id_model.append(row)
         
        print('Selected Leaderboard Model: ', list_id_model[0])
            
        m = h2o.get_model(list_id_model[0]) 
        va_imp=m.varimp(use_pandas=True)
        # Select relevant columns based on the specified relevance percentage.
        n=0.015
        fimp = va_imp[va_imp['percentage'] > n]
        sum_va_imp=fimp['percentage'].sum()
        for iteration in range(0,10): 
            if sum_va_imp<=relevance:
                fimp = va_imp[va_imp['percentage'] > n]
                n=n*0.5
                sum_va_imp=fimp['percentage'].sum()
            elif sum_va_imp>relevance:
                print('Approximated minimum value of Relative Percentage:',n)
                break
        for rows in fimp['variable']: sel_cols.append(rows)  
        
        print('Total relative importance percentage of the selected columns: ', round(sum_va_imp*100,4),'%')
        
        if len(sel_cols)>=5:
            list_t5_cols=sel_cols[0:5]
            print('Top 5 Most Important Input Columns: ', list_t5_cols)
        sel_cols.append(self.target)
        # Return the selected columns and their relative importance percentages.
        
        return sel_cols, fimp

