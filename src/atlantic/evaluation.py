import pandas as pd
from atlantic.analysis import Analysis
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import *
import xgboost as xgb
import optuna
from tqdm import tqdm
import warnings
import logging

class Evaluation(Analysis):
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame, target:str):
        # Constructor for the Evaluation class, inherits from Analysis and initializes class attributes.
        super().__init__(target)
        self.train=train
        self.test=test
        self.metrics=None
        self._tmetrics=None
        self.hparameters_list,self.metrics_list=[],[]
        self.pred_type,self.eval_metric=super().target_type(X=train)
                
    def objective(self,trial,dim:str="normal"): 
        
        X_train,X_test,y_train,y_test=super().divide_dfs(self.train,self.test)
        
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
            "min_samples_split": trial.suggest_int("rf_regressor_min_samples_split", 2, 20),
        }
        
        # Define hyperparameters for Random Forest classification
        rf_classifier_params = {
            "n_estimators": trial.suggest_int("rf_classifier_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("rf_classifier_max_depth", 10, 50),
            "min_samples_split": trial.suggest_int("rf_classifier_min_samples_split", 2, 20),
        }
        
        # Define hyperparameters for Extra Trees regression
        et_regressor_params = {
            "n_estimators": trial.suggest_int("et_regressor_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("et_regressor_max_depth", 5, 32),
            "min_samples_split": trial.suggest_int("et_regressor_min_samples_split", 2, 20),
        }
        
        # Define hyperparameters for Extra Trees classification
        et_classifier_params = {
            "n_estimators": trial.suggest_int("et_classifier_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("et_classifier_max_depth", 10, 50),
            "min_samples_split": trial.suggest_int("et_classifier_min_samples_split", 2, 20),
            }
        
        # Define hyperparameters for XGBoost regression
        xgb_regressor_params = {
            "n_estimators": trial.suggest_int("xgb_regressor_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("xgb_regressor_max_depth", 5, 10),
            "learning_rate": trial.suggest_loguniform("xgb_regressor_learning_rate", 0.01, 0.1),
        }
        
        # Define hyperparameters for XGBoost classification
        xgb_classifier_params = {
            "n_estimators": trial.suggest_int("xgb_classifier_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("xgb_classifier_max_depth", 10, 20),
            "learning_rate": trial.suggest_loguniform("xgb_classifier_learning_rate", 0.05, 0.1),  
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
                    
            if self.pred_type=="Reg":
                # Initialize the regression models with the suggested hyperparameters
                rf_regressor.set_params(**rf_regressor_params)
                if dim!="high":et_regressor.set_params(**et_regressor_params)
                xgb_regressor.set_params(**xgb_regressor_params)
                
                # Train the regression models on the training data
                rf_regressor.fit(X_train, y_train)
                if dim!="high":et_regressor.fit(X_train, y_train)
                xgb_regressor.fit(X_train, y_train)
                
                # Make predictions on the test data for regression models
                rf_r_preds = rf_regressor.predict(X_test)
                if dim!="high":et_r_preds = et_regressor.predict(X_test)
                xgb_r_preds = xgb_regressor.predict(X_test)
                
                m_reg=pd.DataFrame()
                m_rf=pd.DataFrame(self.metrics_regression(y_test,rf_r_preds),index=[0])
                if dim!="high":
                    m_et=pd.DataFrame(self.metrics_regression(y_test,et_r_preds),index=[0])
                    m_et["Model"]="ExtraTrees"
                m_xgb=pd.DataFrame(self.metrics_regression(y_test,xgb_r_preds),index=[0])
                m_rf["Model"],m_xgb["Model"]="RandomForest","XGBoost"
                if dim!="high":m_reg=pd.concat([m_rf,m_et,m_xgb])
                elif dim=="high":m_reg=pd.concat([m_rf,m_xgb])
                m_reg["iteration"]=len(self.metrics_list) + 1
                self.metrics_list.append(m_reg)
                
                self.hparameters_list.append({
                    "rf_regressor_params": rf_regressor_params,
                    "et_regressor_params": et_regressor_params,
                    "xgb_regressor_params": xgb_regressor_params,
                    "iteration": len(self.metrics_list) + 1,
                })
                
            elif self.pred_type=="Class":
                # Initialize the classification model with the suggested hyperparameters
                rf_classifier.set_params(**rf_classifier_params)
                if dim!="high":et_classifier.set_params(**et_classifier_params)
                xgb_classifier.set_params(**xgb_classifier_params)
                
                # Train the classification model on the training data
                rf_classifier.fit(X_train, y_train)
                if dim!="high":et_classifier.fit(X_train, y_train)
                xgb_classifier.fit(X_train, y_train)

                # Make predictions on the test data for classification model
                rf_c_preds = rf_classifier.predict(X_test)
                if dim!="high":et_c_preds = et_classifier.predict(X_test)
                xgb_c_preds = xgb_classifier.predict(X_test)
                
                m_class=pd.DataFrame()
                m_rf=pd.DataFrame(self.metrics_classification(y_test,rf_c_preds),index=[0])
                if dim!="high":
                    m_et=pd.DataFrame(self.metrics_classification(y_test,et_c_preds),index=[0])
                    m_et["Model"]="ExtraTrees"
                m_xgb=pd.DataFrame(self.metrics_classification(y_test,xgb_c_preds),index=[0])
                m_rf["Model"],m_xgb["Model"]="RandomForest","XGBoost"
                if dim!="high":m_class=pd.concat([m_rf,m_et,m_xgb])
                elif dim=="high":m_class=pd.concat([m_rf,m_xgb])
                m_class["iteration"]=len(self.metrics_list)+1
                self.metrics_list.append(m_class)
                
                self.hparameters_list.append({
                    "rf_classifier_params": rf_classifier_params,
                    "et_classifier_params": et_classifier_params,
                    "xgb_classifier_params": xgb_classifier_params,
                    "iteration": len(self.metrics_list)+1,
                    })
        
    def auto_evaluate(self):
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        if len(self.train) <= 8000 and len(list(self.train.columns)) < 30: dim_, n_trials = "low", 8
        elif len(self.train) <= 8000 and len(list(self.train.columns)) >= 30: dim_, n_trials = "medium", 6
        elif len(self.train) > 8000 and len(list(self.train.columns)) < 30: dim_, n_trials = "mid_high", 5
        elif len(self.train) > 8000 and len(list(self.train.columns)) >= 30: dim_, n_trials = "high", 5 # & Reduced Model Ensembles
    
        if self.pred_type=="Reg":
            # Create an Optuna study
            reg_study = optuna.create_study(direction="minimize", study_name="Reg Evaluation")
            # Optimize the objective function with tqdm progress bar
            with tqdm(total=n_trials, desc="",ncols=75) as pbar:
                def trial_callback(study, trial):
                    pbar.update(1)
                reg_study.optimize(lambda trial: self.objective(trial,dim=dim_),
                                   n_trials=n_trials,
                                   callbacks=[trial_callback])
            
            self.metrics=pd.concat(self.metrics_list)
            self.metrics=self.metrics.sort_values(["Model","Mean Absolute Error"], ascending=True)
            self._tmetrics=self.metrics.copy()
            self.metrics=self.metrics.groupby("Model").first().mean(axis=0).to_frame().T
            del self.metrics['iteration']
            
        elif self.pred_type=="Class":
            # Create an Optuna study for classification
            class_study = optuna.create_study(direction="maximize", study_name="Class Evaluation")
            # Optimize the objective function with tqdm progress bar
            with tqdm(total=n_trials, desc="",ncols=75) as pbar:
                def trial_callback(study, trial):
                    pbar.update(1)
                class_study.optimize(lambda trial: self.objective(trial,dim=dim_),
                                     n_trials=n_trials,
                                     callbacks=[trial_callback])
                
            self.metrics=pd.concat(self.metrics_list)
            self.metrics=self.metrics.sort_values(["Model","Accuracy"], ascending=False)
            self._tmetrics=self.metrics.copy()
            self.metrics=self.metrics.groupby("Model").first().mean(axis=0).to_frame().T
            del self.metrics['iteration']
        
        return self.metrics 
    
    @staticmethod
    def metrics_regression(y_true, y_pred):
        # Calculate various regression model evaluation metrics.
        mae=mean_absolute_error(y_true, y_pred)
        mape=mean_absolute_percentage_error(y_true, y_pred)
        mse=mean_squared_error(y_true, y_pred)
        evs=explained_variance_score(y_true, y_pred)
        maximo_error=max_error(y_true, y_pred)
        r2=r2_score(y_true, y_pred)
        metrics_reg= {'Mean Absolute Error': mae,
                      'Mean Absolute Percentage Error': mape,
                      'Mean Squared Error': mse,
                      'Explained Variance Score': evs,
                      'Max Error': maximo_error,
                      'R2 Score':r2}
        
        return metrics_reg
    
    @staticmethod
    def metrics_classification(y_true, y_pred):
        # Calculate various classification model evaluation metrics.
        accuracy_metric=accuracy_score(y_true, y_pred)
        precision_metric=precision_score(y_true, y_pred,average='micro')
        f1_macro_metric=f1_score(y_true, y_pred,average='macro')
        recall_score_metric=recall_score(y_true, y_pred, average='macro')
        
        metrics_class={'Accuracy': accuracy_metric,
                       'Precision Micro': precision_metric,
                       'F1 Score Macro':f1_macro_metric,
                       'Recall Score Macro':recall_score_metric}
        
        return metrics_class





