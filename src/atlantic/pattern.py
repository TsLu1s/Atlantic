import pandas as pd
from atlantic.evaluation import Evaluation
from atlantic.imputation import AutoSimpleImputer, AutoKNNImputer, AutoIterativeImputer
from atlantic.processing import Encoding_Version
from atlantic.selector import Selector

class Pattern(Evaluation):
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame, target:str):
        super().__init__(train,test,target)
        self.enc_method=None
        self.imp_method=None
        self._imputer=None
        self.perf=None
        
    def encoding_selection(self):
        # Select the encoding method and assess its performance.
        
        # Create an Encoding_Version instance and a SimpleImputer instance.
        ev = Encoding_Version(train=self.train.copy(), test=self.test.copy(), target=self.target)
        simple_imputer = AutoSimpleImputer(strategy='mean',target=self.target)
        
        # Generate different encoding versions and separate train and test data for each encoding version.
        ds = [method() for method in [ev.encoding_v1, ev.encoding_v2, ev.encoding_v3, ev.encoding_v4]]
        (train_v1, test_v1), (train_v2, test_v2), (train_v3, test_v3), (train_v4, test_v4) = [
            (d[0], d[1]) if len(d) == 2 else (None, None) for i, d in enumerate(ds)]
            
        # Impute missing values if necessary.
        if (train_v1.isnull().sum().sum() or test_v1.isnull().sum().sum()) != 0:
            train_v1, test_v1 = simple_imputer.impute(train=train_v1,test=test_v1)
            
        if (train_v2.isnull().sum().sum() or test_v2.isnull().sum().sum()) != 0:
            train_v2, test_v2 = simple_imputer.impute(train=train_v2,test=test_v2)
            
        if (train_v3.isnull().sum().sum() or test_v3.isnull().sum().sum()) != 0:
            train_v3, test_v3 = simple_imputer.impute(train=train_v3,test=test_v3)
            
        if (train_v4.isnull().sum().sum() or test_v4.isnull().sum().sum()) != 0:
             train_v4, test_v4 = simple_imputer.impute(train=train_v4,test=test_v4)
            
        # Perform model evaluation for each encoding version.
        p_v1 = Evaluation(train=train_v1,
                          test=test_v1,
                          target=self.target).pred_eval()[self.eval_metric][0]
        
        p_v2 = Evaluation(train=train_v2,
                          test=test_v2,
                          target=self.target).pred_eval()[self.eval_metric][0]
        
        p_v3 = Evaluation(train=train_v3,
                          test=test_v3,
                          target=self.target).pred_eval()[self.eval_metric][0]
        
        p_v4 = Evaluation(train=train_v4,
                          test=test_v4,
                          target=self.target).pred_eval()[self.eval_metric][0]
        
        if self.pred_type=='Reg':
            print(' ')
            print('Predictive Performance Encoding Versions:')
            print('\n MAE Version 1 [IDF + StandardScaler] : ', round(p_v1, 4),
                  '\n MAE Version 2 [IDF + MinMaxScaler] : ', round(p_v2, 4),
                  '\n MAE Version 3 [Label + StandardScaler] : ', round(p_v3, 4),
                  '\n MAE Version 4 [Label + MinMaxScaler] : ', round(p_v4, 4))
            metric='MAE'
        elif self.pred_type=='Class':
            print(' ')
            print('Predictive Performance Encoding Versions:')
            print('\n ACC Version 1 [IDF + StandardScaler] : ', round(p_v1, 4),
                  '\n ACC Version 2 [IDF + MinMaxScaler] : ', round(p_v2, 4),
                  '\n ACC Version 3 [Label + StandardScaler] : ', round(p_v3, 4),
                  '\n ACC Version 4 [Label + MinMaxScaler] : ', round(p_v4, 4))
            metric='ACC'
        
        list_encoding=[p_v1,p_v2,p_v3,p_v4]
        list_encoding.sort()
        if self.pred_type=='Class': list_encoding.sort(reverse=True)
        
        # Select the encoding method with the best performance.
        if list_encoding[0]==p_v1:
            self.enc_method='Encoding Version 1'
            print('Encoding Version 1 was choosen with an ', metric, ' of: ', round(p_v1, 4))
            
        elif list_encoding[0]==p_v2:
            self.enc_method='Encoding Version 2'
            print('Encoding Version 2 was choosen with an ', metric, ' of: ', round(p_v2, 4))    
            
        elif list_encoding[0]==p_v3:
            self.enc_method='Encoding Version 3'
            print('Encoding Version 3 was choosen with an ', metric, ' of: ', round(p_v3, 4))
            
        elif list_encoding[0]==p_v4:
            self.enc_method='Encoding Version 4'
            print('Encoding Version 4 was choosen with an ', metric, ' of: ', round(p_v4, 4))
            print(' ')
                
        return self.enc_method
        
    def imputation_selection(self):
        
        # Select the imputation method and assess its performance.
        if self.pred_type=='Reg': metric='MAE' 
        elif self.pred_type=='Class': metric='ACC'
        
        ev = Encoding_Version(train=self.train.copy(), test=self.test.copy(), target=self.target)
        # Depending on the selected encoding method, apply the corresponding encoding function.
        if self.enc_method=='Encoding Version 1':
            self.train, self.test=ev.encoding_v1()
        elif self.enc_method=='Encoding Version 2':
            self.train, self.test=ev.encoding_v2()
        elif self.enc_method=='Encoding Version 3':
            self.train, self.test=ev.encoding_v3()
        elif self.enc_method=='Encoding Version 4':
            self.train, self.test=ev.encoding_v4()
        elif self.enc_method==None:
            print(" No Encoding Version Selected ")
            self.train, self.test=ev.encoding_v4()
            
        print('    ')   
        print('Simple Imputation Loading')
        simple_imputer = AutoSimpleImputer(strategy='mean')
        simple_imputer.fit(self.train)  # Fit on the DataFrame
        train_s = simple_imputer.transform(self.train.copy())  # Transform the Train DataFrame
        test_s = simple_imputer.transform(self.test.copy()) # Transform the Test DataFrame
        
        print('KNN Imputation Loading')
        knn_imputer = AutoKNNImputer(n_neighbors=3) 
        knn_imputer.fit(self.train)  # Fit on the DataFrame
        train_knn = knn_imputer.transform(self.train.copy())  # Transform the Train DataFrame
        test_knn = knn_imputer.transform(self.test.copy()) # Transform the Test DataFrame
        
        print('Iterative Imputation Loading')
        iter_imputer = AutoIterativeImputer(max_iter=10, random_state=42)
        iter_imputer.fit(self.train)  # Fit on the DataFrame
        train_iter = iter_imputer.transform(self.train.copy())  # Transform the Train DataFrame
        test_iter = iter_imputer.transform(self.test.copy()) # Transform the Test DataFrame
        
        p_s = Evaluation(train=train_s,
                         test=test_s,
                         target=self.target).pred_eval()[self.eval_metric][0]
        
        p_knn = Evaluation(train=train_knn,
                           test=test_knn,
                           target=self.target).pred_eval()[self.eval_metric][0]
        
        p_iter = Evaluation(train=train_iter,
                            test=test_iter,
                            target=self.target).pred_eval()[self.eval_metric][0]
            
        print('    ')
        print('Predictive Performance Null Imputation Versions:')
        
        print(' KNN Performance: ', round(p_knn, 4), '\n Iterative Performance: ', 
              round(p_iter, 4),'\n Simple Performance: ', round(p_s, 4))
        
        list_imp=[p_iter,p_knn,p_s]
        list_imp.sort()
        if self.pred_type=='Class': list_imp.sort(reverse=True)
        
        # Select the imputation method with the best performance.
        if list_imp[0]==p_s:
            self.imp_method='Simple'  
            self.train, self.test,self._imputer=train_s.copy(),test_s.copy(),simple_imputer
            print('Simple Imputation Algorithm was chosen with an ', metric, ' of: ', round(p_s, 4))
            
        elif list_imp[0]==p_knn:
            self.imp_method='KNN'
            self.train, self.test,self._imputer=train_knn.copy(), test_knn.copy(),knn_imputer
            print('KNN Imputation Algorithm was chosen with an ', metric, ' of: ', round(p_knn, 4))
            
        elif list_imp[0]==p_iter:
            self.imp_method='Iterative'
            self.train, self.test,self._imputer=train_iter.copy(),test_iter.copy(),iter_imputer
            print('Iterative Imputation Algorithm was chosen with an ', metric, ' of: ', round(p_iter, 4))
        self.perf=list_imp[0]
        
        return self.train, self.test
    
    def vif_performance(self,vif_threshold:float=10.0,perf_:float=None):
        # Perform VIF-based feature selection on the train and test data.
        # Evaluate and compare the performance before and after VIF-based feature selection.
        train_vif, test_vif = self.train.copy(), self.test.copy()

        cols_vif = Selector(X=train_vif,target=self.target).feature_selection_vif(vif_threshold=vif_threshold)
        print('    ')
        print('Number of Selected VIF Columns: ', len(cols_vif),
              '\nRemoved Columns with VIF :', len(list(train_vif.columns)) - len(cols_vif))
        
        train_vif,test_vif = train_vif[cols_vif],test_vif[cols_vif]
        
        if perf_!=None: self.perf=perf_
        if self.perf==None:
            self.perf = Evaluation(train=self.train.copy(),
                                     test=self.test.copy(),
                                     target=self.target).pred_eval()[self.eval_metric][0]
            
        perf_vif = Evaluation(train=train_vif,
                              test=test_vif,
                              target=self.target).pred_eval()[self.eval_metric][0]
        
        print('Default Performance:', round(self.perf, 4))
        print('VIF Performance:', round(perf_vif, 4))
    
        if self.pred_type == 'Reg':
            if perf_vif < self.perf:
                print('The VIF filtering method was applied')
                self.train = self.train[cols_vif]
                self.test = self.test[cols_vif]
            else:
                print('The VIF filtering method was not applied')
        elif self.pred_type == 'Class':
            if perf_vif > self.perf:
                print('The VIF filtering method was applied')
                self.train = self.train[cols_vif]
                self.test = self.test[cols_vif]
            else:
                print('The VIF filtering method was not applied')
    
        return self.train, self.test
    

    