import pandas as pd
from atlantic.processing.analysis import Analysis
from atlantic.optimizer.metrics import metrics_classification, metrics_regression
from sklearn.ensemble import (RandomForestRegressor,
                              ExtraTreesRegressor,
                              RandomForestClassifier,
                              ExtraTreesClassifier)
import xgboost as xgb
import optuna
from tqdm import tqdm
import warnings
import logging

class Evaluation(Analysis):
    def __init__(self, 
                 train : pd.DataFrame,
                 test : pd.DataFrame,
                 target : str):
        """
        Performs hyperparameter optimization and model evaluation for both regression and classification tasks.
        It extends the Analysis class, leveraging its methods for data handling and prediction type identification.
        
        The main objective of the Evaluation class is to automate the process of selecting optimal 
        hyperparameters for different machine learning models using the Optuna optimization framework. 
        This class supports models such as Random Forest, Extra Trees, and XGBoost, and provides a 
        systematic approach to evaluating their performance on a given dataset.
    
        Methods
        -------
        objective(trial, dim: str = "normal"):
            Defines the objective function for hyperparameter optimization using Optuna. It configures 
            the hyperparameters for various models, trains them, and evaluates their performance.
            
        auto_evaluate():
            Automates the evaluation process by determining the appropriate optimization settings based 
            on the dataset size and dimensions. It orchestrates the optimization trials and aggregates 
            the evaluation metrics for comparison.
        """
        # Constructor for the Evaluation class, inherits from Analysis and initializes class attributes.
        super().__init__(target)
        self.train = train
        self.test = test
        self.metrics = None
        self._tmetrics = None
        self.hparameters_list, self.metrics_list = [], []
        self.pred_type, self.eval_metric = super().target_type(X = train)
                
    def objective(self,
                  trial,
                  dim : str = "normal"): 
        """
        Defines the objective function for hyperparameter optimization using Optuna.
        
        This method configures the hyperparameters for RandomForest, ExtraTrees, and XGBoost models 
        for both regression and classification tasks. It trains these models on the training set and 
        evaluates their performance on the test set using appropriate metrics. The results of each 
        trial are stored for further analysis and comparison.

        Parameters:
        - trial: optuna.trial.Trial
            An Optuna trial object used to suggest hyperparameter values.
        - dim: str, optional (default = "normal")
            A string indicating the dimensionality setting for the models, which can affect the 
            hyperparameter configuration.

        Returns:
        - None
        """
        
        X_train, X_test, y_train, y_test = super().divide_dfs(self.train,self.test)
        
        # Configure logging to suppress Optuna's logs
        logging.getLogger('optuna').setLevel(logging.CRITICAL)
        
        # Define the regression and classification models
        rf_regressor = RandomForestRegressor()
        et_regressor = ExtraTreesRegressor()
        xgb_regressor = xgb.XGBRegressor()

        rf_classifier = RandomForestClassifier()
        et_classifier = ExtraTreesClassifier()
        xgb_classifier = xgb.XGBClassifier()
            
        # Define hyperparameters for Random Forest regression
        rf_regressor_params = {
            "n_estimators": trial.suggest_int("rf_regressor_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("rf_regressor_max_depth", 5, 32),
            "min_samples_split": trial.suggest_int("rf_regressor_min_samples_split", 2, 25),
        }
        
        # Define hyperparameters for Random Forest classification
        rf_classifier_params = {
            "n_estimators": trial.suggest_int("rf_classifier_n_estimators", 60, 250),
            "max_depth": trial.suggest_int("rf_classifier_max_depth", 10, 50),
            "min_samples_split": trial.suggest_int("rf_classifier_min_samples_split", 2, 20),
        }
        
        # Define hyperparameters for Extra Trees regression
        et_regressor_params = {
            "n_estimators": trial.suggest_int("et_regressor_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("et_regressor_max_depth", 5, 32),
            "min_samples_split": trial.suggest_int("et_regressor_min_samples_split", 2, 25),
        }
        
        # Define hyperparameters for Extra Trees classification
        et_classifier_params = {
            "n_estimators": trial.suggest_int("et_classifier_n_estimators", 60, 250),
            "max_depth": trial.suggest_int("et_classifier_max_depth", 10, 50),
            "min_samples_split": trial.suggest_int("et_classifier_min_samples_split", 2, 20),
            }
        
        # Define hyperparameters for XGBoost regression
        xgb_regressor_params = {
            "n_estimators": trial.suggest_int("xgb_regressor_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("xgb_regressor_max_depth", 5, 25),
            "learning_rate": trial.suggest_loguniform("xgb_regressor_learning_rate", 0.01, 0.1),
        }
        
        # Define hyperparameters for XGBoost classification
        xgb_classifier_params = {
            "n_estimators": trial.suggest_int("xgb_classifier_n_estimators", 60, 250),
            "max_depth": trial.suggest_int("xgb_classifier_max_depth", 10, 20),
            "learning_rate": trial.suggest_loguniform("xgb_classifier_learning_rate", 0.05, 0.1),  
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
                    
            if self.pred_type in ["Reg", "Class"]:
                # Define models and parameters
                models = [rf_regressor, et_regressor, xgb_regressor] if self.pred_type == "Reg" else [rf_classifier, et_classifier, xgb_classifier]
                params = [rf_regressor_params, et_regressor_params, xgb_regressor_params] if self.pred_type == "Reg" else [rf_classifier_params, et_classifier_params, xgb_classifier_params]
                metrics_func = metrics_regression if self.pred_type == "Reg" else metrics_classification
            
                # Train models
                for model, param in zip(models, params):
                    if dim != "high" or model != et_regressor:
                        model.set_params(**param)
                        model.fit(X_train, y_train)
            
                # Make predictions
                preds = [model.predict(X_test) if dim != "high" or model != et_regressor else None for model in models]
            
                # Calculate metrics
                metrics = [metrics_func(y_test, pred) if pred is not None else None for pred in preds]
            
                # Concatenate metrics
                m_df = pd.concat([m for m in metrics if m is not None])
            
                # Set model names
                m_df["Model"] = ["RandomForest", "ExtraTrees", "XGBoost"] if dim != "high" else ["RandomForest", "XGBoost"]
            
                # Set iteration
                m_df["iteration"] = len(self.metrics_list) + 1
            
                # Append to metrics list
                self.metrics_list.append(m_df)
            
                # Append hyperparameters
                hparams = {}
                for model_name, param in zip(["rf", "et", "xgb"], params):
                    hparams[model_name + "_" + self.pred_type.lower() + "_params"] = param
                hparams["iteration"] = len(self.metrics_list) + 1
                self.hparameters_list.append(hparams)
    
    def auto_evaluate(self):
        """
        Automates the evaluation process by determining the appropriate optimization settings based 
        on the dataset size and dimensions.
        
        This method orchestrates the optimization trials using Optuna, dynamically adjusting the 
        number of trials and dimensionality settings based on the characteristics of the dataset. 
        It aggregates the evaluation metrics from all trials and identifies the best-performing models.

        Returns:
        - metrics: pd.DataFrame
            A DataFrame containing the aggregated evaluation metrics for the best-performing models.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        train_len, train_cols = self.train.shape[0], self.train.shape[1]
        
        if train_len <= 8000:
            dim_, n_trials = ("low", 8) if train_cols < 30 else ("medium", 6)
        else:
            dim_, n_trials = ("mid_high", 5) if train_cols < 30 else ("high", 5)
    
        study_params = {
            "direction": "minimize" if self.pred_type == "Reg" else "maximize",
            "study_name": f"{self.pred_type} Evaluation"
        }
    
        study = optuna.create_study(**study_params)
    
        with tqdm(total = n_trials, desc = "", ncols = 75) as pbar:
            def trial_callback(study, trial):
                pbar.update(1)
    
            study.optimize(lambda trial: self.objective(trial, dim = dim_), 
                           n_trials = n_trials, 
                           callbacks = [trial_callback])
    
        self.metrics = pd.concat(self.metrics_list)
        sort_col = "Mean Absolute Error" if self.pred_type == "Reg" else "Accuracy"
        self.metrics = self.metrics.sort_values(["Model", sort_col], ascending=self.pred_type == "Reg")
        self._tmetrics = self.metrics.copy()
        self.metrics = self.metrics.groupby("Model").first().mean(axis=0).to_frame().T
        self.metrics.drop(columns = 'iteration', inplace=True)
        
        return self.metrics
    





