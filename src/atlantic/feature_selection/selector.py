import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from statsmodels.stats.outliers_influence import variance_inflation_factor
from atlantic.processing.encoders import AutoLabelEncoder 
from atlantic.processing.analysis import Analysis 

class Selector(Analysis):
    def __init__(self, 
                 X : pd.DataFrame, 
                 target : str):
        super().__init__(target)
        self.X = X
        self.vif_df = None
        self.pred_type,self.eval_metric = super().target_type(X = X)
        """
        The objective of this class is to facilitate the identification of the most relevant features 
        for machine learning models, thereby improving model performance and interpretability. It leverages 
        both statistical techniques, such as Variance Inflation Factor (VIF) for reducing multicollinearity, 
        and advanced machine learning methods, such as H2O AutoML, for selecting features based on their 
        predictive power.
    
        Includes tools for determining the type of prediction task (classification or regression) and the appropriate evaluation 
        metric. The primary methods of the Selector class focus on calculating VIF values and performing feature 
        selection using both VIF and H2O AutoML.
    
        Methods
        -------
        calculate_vif(X: pd.DataFrame):
            Calculates the Variance Inflation Factor (VIF) for numeric columns in the dataset.
            
        feature_selection_vif(vif_threshold: float = 10.0):
            Performs feature selection by iteratively removing features with VIF values above a specified threshold.
            
        feature_selection_h2o(relevance: float = 0.99, h2o_fs_models: int = 7, encoding_fs: bool = True):
            Performs feature selection using H2O AutoML, selecting features based on their importance scores and a specified relevance threshold.
        """
        
    def calculate_vif(self,X : pd.DataFrame):
        """
        Calculate Variance Inflation Factor (VIF) for numeric columns in the dataset.
        
        The Variance Inflation Factor (VIF) is a measure used to detect the presence of multicollinearity among 
        numerical features. High VIF values indicate that a feature is highly collinear with other features, 
        which can adversely affect the performance and interpretability of machine learning models. This method 
        computes the VIF for each numerical feature in the dataset, helping identify and mitigate multicollinearity.

        Parameters:
        - X: pd.DataFrame
            The dataset containing the features for which VIF will be calculated. It should only contain 
            numerical columns, as VIF is not applicable to categorical columns.

        Returns:
        - vif: pd.DataFrame
            A DataFrame containing the VIF values for each feature, sorted in descending order of VIF values.

        Raises:
        - ValueError: If the dataset contains non-numerical columns or null values.
        """
        
        # Check if there are any categorical columns or null values in X
        if len([col for col in X[list(X.columns)].select_dtypes(include=['number']).columns if col != self.target]) < len(list(X.columns))-1: 
            raise ValueError("Only numerical columns are supported in VIF calculation.")
        if X.isnull().values.any():
            raise ValueError("Null values are not supported in VIF calculation.")

        vif = pd.DataFrame()
        vif['variables'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif.sort_values(['VIF'], ascending = False)
        
        return vif

    def feature_selection_vif(self, vif_threshold : float = 10.0):
        """
        Perform feature selection using VIF (Variance Inflation Factor).
        
        This method iteratively removes features with high VIF values (above a specified threshold) to reduce 
        multicollinearity in the dataset. By removing features with high multicollinearity, the method aims to 
        improve the robustness and interpretability of the resulting models. This process helps in retaining 
        features that contribute unique information to the model, enhancing its performance.

        Parameters:
        - vif_threshold: float
            The threshold above which features with high VIF values will be removed. The default value is 10.0.
            It must be within the interval [3, 30].

        Returns:
        - cols: list
            The list of selected feature columns after VIF-based feature selection, including the target column.

        Raises:
        - AssertionError: If the vif_threshold is not within the interval [3, 30].
        """
        # Perform feature selection using VIF (Variance Inflation Factor).
        assert vif_threshold >= 3 and vif_threshold <= 30, 'VIF threshold should be in [3, 30] interval'
        
        cols = list(self.X.columns.difference([self.target]))
        X_ = self.X[cols].copy()
        self.vif_df = self.calculate_vif(X_)

        while self.vif_df['VIF'].max() >= vif_threshold:
            # Iteratively remove columns with VIF above the threshold.
            self.vif_df.drop(self.vif_df['variables'].loc[self.vif_df['VIF'] == self.vif_df['VIF'].max()].index,
                             inplace = True)
            cols = [rows for rows in self.vif_df['variables']]
            X_ = X_[cols]
            self.vif_df = self.calculate_vif(X_)
        cols.append(self.target)
        
        return cols
        
    def feature_selection_h2o(self,
                              relevance : float = 0.99,
                              h2o_fs_models : int = 7,
                              encoding_fs : bool = True):
        """
        Perform feature selection using H2O AutoML and relevance percentage.
        
        This method leverages H2O's AutoML to identify and select the most relevant features for the prediction 
        task. It trains multiple models and uses their feature importances to determine which features should 
        be retained based on a specified relevance threshold. By utilizing H2O AutoML, this method can automate 
        the feature selection process, ensuring that the most predictive features are identified.

        Parameters:
        - relevance: float
            The cumulative relevance threshold for selecting features based on their importance scores. It must
            be within the interval [0.4, 1]. The default value is 0.99.
        - h2o_fs_models: int
            The number of models to be trained by H2O AutoML for feature selection. It must be within the 
            interval [1, 100]. The default value is 7.
        - encoding_fs: bool
            A flag indicating whether to apply label encoding to categorical features before training the models.
            The default value is True.

        Returns:
        - sel_cols: list
            The list of selected feature columns after H2O AutoML-based feature selection, including the target column.
        - fimp: pd.DataFrame
            A DataFrame containing the selected features and their importance scores.

        Raises:
        - AssertionError: If the relevance or h2o_fs_models parameters are not within their specified intervals.
        """
        # Perform feature selection using H2O AutoML and relevance percentage.
        assert relevance >= 0.4 and relevance <= 1 , 'relevance value should be in [0.4,1] interval'
        assert h2o_fs_models >= 1 and h2o_fs_models <= 100 , 'h2o_fs_models value should be in [0,100] interval'
    
        # Initialize H2O and prepare the data for feature selection.
        h2o.init()
        
        X_ = self.X.copy()
        
        if encoding_fs == True:
            encoder = AutoLabelEncoder()
            encoder.fit(X_[[col for col in X_.select_dtypes(include=['object','category']).columns if col != self.target]])
            X_ = encoder.transform(X=X_)
            
        train_h2o = h2o.H2OFrame(X_)

        if self.pred_type == 'Class': 
            train_h2o[self.target] = train_h2o[self.target].asfactor()
        # Train an AutoML model and retrieve the leaderboard.
        aml = H2OAutoML(max_models = h2o_fs_models,
                        nfolds = 3,
                        seed = 1,
                        exclude_algos = ['GLM',  
                                         'DeepLearning', 
                                         'StackedEnsemble'],
                        sort_metric = 'AUTO')
        aml.train(x = list(X_.columns.difference([self.target])),
                  y = self.target,
                  training_frame = train_h2o)
        leaderboards_df = aml.leaderboard.as_data_frame()
        print(leaderboards_df)
        
        list_id_model = leaderboards_df['model_id'].tolist()
         
        print('Leaderboard Feature Selection Model: ', list_id_model[0])
            
        m = h2o.get_model(list_id_model[0]) 
        va_imp = m.varimp(use_pandas = True)
        
        # Close H2O session
        h2o.shutdown(prompt = False)  

        # Select relevant columns based on the specified relevance percentage.
        n = 0.015
        fimp = va_imp[va_imp['percentage'] > n]
        sum_va_imp = fimp['percentage'].sum()
        for _ in range(0,10): 
            if sum_va_imp <= relevance:
                fimp = va_imp[va_imp['percentage'] > n]
                n = n*0.5
                sum_va_imp = fimp['percentage'].sum()
            elif sum_va_imp > relevance:
                break
        sel_cols = fimp['variable'].tolist()
        sel_cols.append(self.target)
        # Return the selected columns and their relative importance percentages.
        
        return sel_cols, fimp

