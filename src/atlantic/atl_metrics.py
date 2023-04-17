import numpy as np
import pandas as pd
from sklearn.metrics import *

def metrics_regression(y_real, y_pred): 
    """
    Function to calculate various regression model evaluation metrics
    Parameters:
    y_real: array-like, shape = (n_samples)
            Real values of the target
    y_pred: array-like, shape = (n_samples)
            Predicted values of the target
    Returns:
    metrics_reg: dictionary with keys as the metrics names, and values as the respective values
    """
    mae=mean_absolute_error(y_real, y_pred)
    mape= mean_absolute_percentage_error(y_real, y_pred)
    mse=mean_squared_error(y_real, y_pred)
    evs= explained_variance_score(y_real, y_pred)
    maximo_error= max_error(y_real, y_pred)
    r2=r2_score(y_real, y_pred)
    metrics_reg= {'Mean Absolute Error': mae, 
                  'Mean Absolute Percentage Error': mape,
                  'Mean Squared Error': mse,
                  'Explained Variance Score': evs, 
                  'Max Error': maximo_error,
                  'R2 Score':r2}  
    
    return metrics_reg

def metrics_classification(y_true, y_pred):

    accuracy_metric = accuracy_score(y_true, y_pred)
    precision_metric = precision_score(y_true, y_pred,average='micro')
    f1_macro_metric = f1_score(y_true, y_pred,average='macro')
    
    metrics_class= {'Accuracy': accuracy_metric,
                    'Precision Micro': precision_metric,
                    'F1 Score Macro':f1_macro_metric,
                    }
    
    return metrics_class
