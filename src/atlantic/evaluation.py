import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import *
from atlantic.analysis import Analysis

class Evaluation(Analysis):
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame, target:str):
        # Constructor for the Evaluation class, inherits from Analysis and initializes class attributes.
        super().__init__(target)
        self.train=train
        self.test=test
        self.metric=None
        self.pred_type,self.eval_metric=super().target_type(X=train)
        
    def pred_eval(self):
        # Perform model evaluation for regression or classification tasks.
        X_train,X_test,y_train,y_test=super().divide_dfs(self.train,self.test)
        
        list_estimators,rf,et=[100,150],[],[]
        
        for estimators in list_estimators:
            
            if self.pred_type=='Reg': 
                # For regression tasks
                regressor_RF = RandomForestRegressor(n_estimators=estimators, random_state=42)
                regressor_RF.fit(X_train, y_train)
                y_pred_rfr = regressor_RF.predict(X_test)
                RF_perf=pd.DataFrame(self.metrics_regression(y_test,y_pred_rfr),index=[0])
                RF_perf[['Estimators']]=estimators
                rf.append(RF_perf)
                        
                Reg_ET = ExtraTreesRegressor(n_estimators=estimators, random_state=42)
                Reg_ET.fit(X_train, y_train)
                y_pred_etr = Reg_ET.predict(X_test)
                ET_perf=pd.DataFrame(self.metrics_regression(y_test,y_pred_etr),index=[0])
                ET_perf[['Estimators']]=estimators
                et.append(ET_perf)
                a,b=pd.concat(rf),pd.concat(et)
                x=pd.concat([a,b]) 
                x=x.sort_values(self.eval_metric, ascending=True)
                self.metric=x.copy()
                
            elif self.pred_type=='Class': 
                # For classification tasks
                classifier_RF = RandomForestClassifier(n_estimators=estimators, random_state=42)
                classifier_RF.fit(X_train, y_train)
                y_pred_rfc = classifier_RF.predict(X_test)
                RF_perf=pd.DataFrame(self.metrics_classification(y_test,y_pred_rfc),index=[0])
                RF_perf[['Estimators']]=estimators
                rf.append(RF_perf)
                
                classifier_ET = ExtraTreesClassifier(n_estimators=estimators, random_state=42)
                classifier_ET.fit(X_train, y_train)
                y_pred_etc = classifier_ET.predict(X_test)
                ET_perf=pd.DataFrame(self.metrics_classification(y_test,y_pred_etc),index=[0])
                ET_perf[['Estimators']]=estimators
                et.append(ET_perf)
                a,b=pd.concat(rf),pd.concat(et)
                x=pd.concat([a,b]) 
                x=x.sort_values(self.eval_metric, ascending=False)
                self.metric=x.copy()

        del x['Estimators']
        metrics=(x.iloc[:1,:]+x.iloc[1:2,:])/2
        
        return metrics
        
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

