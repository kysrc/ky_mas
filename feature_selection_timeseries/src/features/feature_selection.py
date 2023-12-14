# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the feature selection method(s) for computing feature importances

import numpy as np
import sage
import xgboost as xgb
import random
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
from sklearn.metrics import accuracy_score 
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
import matplotlib.pyplot as plt
import shap
import torch
import math
from boruta import BorutaPy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchmetrics import Accuracy, AUROC
from feature_selection_timeseries.author_feature_selection.dynamic_selection_main import dynamic_selection as ds
from feature_selection_timeseries.author_feature_selection.dynamic_selection_main.dynamic_selection import MaskingPretrainer, GreedyDynamicSelection
from feature_selection_timeseries.author_feature_selection.stg_master.python.stg.stg import STG
from feature_selection_timeseries.src.models.utils import setup_seed


class featureValues:
    """ A class containing feature selection methods
    Args:
        data_dict (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        model (obj): a trained XGBoost model
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
        n_features (int): number of features to use; used for top n and bottom n features
    """
    def __init__(self, print_outputs_train, params, data_dict, pred_type, model, seed, target_colname, n_features):
        self.print_outputs_train = print_outputs_train
        self.params = params
        self.X_data_train = data_dict["X_train"]
        self.X_data_val = data_dict["X_val"]
        self.y_data_train = data_dict["y_train"]
        self.y_data_val = data_dict["y_val"]
        self.pred_type = pred_type.lower()
        self.model = model
        self.sage_val = None
        self.feature_names = self.X_data_train.columns.to_list()
        self.seed = seed
        self.model_init = xgb.XGBClassifier(**params, objective='binary:logistic', eval_metric='error', seed=self.seed) if self.pred_type.lower() == 'classification' else xgb.XGBRegressor(**params, objective='reg:squarederror', eval_metric='rmse', seed=self.seed)
        self.n_features = n_features
        setup_seed(self.seed)
    
    def stg_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Paper: https://arxiv.org/abs/1810.04247
            Source: https://github.com/runopti/stg/tree/master
            Compiled_by: JYang
        Returns: 
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """  
        start_time = time.time()
        
        args_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if args_cuda else "cpu") 
        model = STG(
            task_type = 'classification' if self.pred_type == "classification" else 'regression',
            input_dim = self.X_data_train.shape[1], 
            output_dim = 2 if self.pred_type == "classification" else 1,
            hidden_dims = [60, self.X_data_train.shape[1]] if self.pred_type == "classification" else [500, 50, 10],  
            activation = 'tanh',
            optimizer = 'SGD', 
            learning_rate = 0.1, 
            batch_size = self.X_data_train.shape[0], 
            feature_selection = True, 
            sigma = 0.5, 
            lam = 0.5, 
            random_state = 1, 
            device = device
        ) 

        model.fit(
            X=self.X_data_train.values, 
            y=np.array([self.y_data_train.values]).reshape(-1, 1),  # reshaped to avoid the shape warning
            nr_epochs=10000, 
            valid_X=self.X_data_val.values, 
            valid_y=np.array([self.y_data_val.values]).reshape(-1, 1), # reshaped to avoid the shape warning
            print_interval=1000#,
            #early_stop="True"
        )
        
        feature_impt = model.get_gates(mode='raw')
        feature_rank_index = np.argsort(-feature_impt)
        feature_names_sorted = np.array(self.feature_names)[feature_rank_index].tolist()
        feature_scores = feature_impt[feature_rank_index].tolist()

        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        
        return feature_names_sorted, feature_scores, total_time
    
    def dynamic_selection_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Paper: https://arxiv.org/abs/2301.00557
            Source: https://github.com/iancovert/dynamic-selection/tree/main
            Compiled_by: JYang
        Returns: 
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """  
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert Pandas DataFrame to a PyTorch Tensor
        train_features = torch.tensor(self.X_data_train.values, dtype=torch.float32)
        train_targets = torch.tensor(self.y_data_train.values, dtype=torch.long)
        val_features = torch.tensor(self.X_data_val.values, dtype=torch.float32)
        val_targets = torch.tensor(self.y_data_val.values, dtype=torch.long)
        # Create a TensorDataset
        train_ds = TensorDataset(train_features, train_targets)
        val_ds = TensorDataset(val_features, val_targets)
        # Prepare dataloaders
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=False, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1024, pin_memory=False)
        # Shape of input and output
        d_in = self.X_data_train.shape[1]
        d_out = len(set(self.y_data_train))
        
        # Set up networks
        hidden = 128
        dropout = 0.3

        predictor = nn.Sequential(
            nn.Linear(2 * d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out))

        selector = nn.Sequential(
            nn.Linear(2 * d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_in))
        
        # Pretrain predictor.
        mask_layer = ds.utils.MaskLayer(append=True)
        pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
        pretrain.fit(
            train_loader,
            val_loader,
            lr = 1e-3,
            nepochs = 100,
            loss_fn = nn.CrossEntropyLoss(),
            verbose = False)

        # Train selector and predictor jointly.
        gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
        gdfs.fit(
            train_loader,
            val_loader,
            lr = 1e-3,
            nepochs = 50,
            max_features = d_in,
            loss_fn = nn.CrossEntropyLoss(),
            verbose = False)
        
        # For saving results.
        num_features = np.arange(1, d_in).tolist() 
        auroc_list = []
        acc_list = []

        # Metrics (softmax is applied automatically in recent versions of torchmetrics)
        auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
        acc_metric = Accuracy(task='multiclass', num_classes=d_out)
        
        # Evaluate.
        for num in num_features:
            #auroc, acc, feature_rank = gdfs.evaluate(test_loader, num, (auroc_metric, acc_metric))
            score, feature_rank = gdfs.evaluate(val_loader, num, (auroc_metric, acc_metric))
            auroc_list.append(score[0])
            acc_list.append(score[1])
            #print(f'Num = {num}, AUROC = {100*score[0]:.2f}, Acc = {100*score[1]:.2f}')
            
        # Compute feature rank
        feature_rank_index = np.argsort(np.mean(feature_rank, axis = 0))
        #feature_score_unordered = np.sort(np.mean(feature_rank, axis = 0)).tolist()
        
        feature_names_sorted = np.array(self.feature_names)[feature_rank_index].tolist()
        feature_scores = np.arange(len(self.feature_names), 0, -1).tolist()
        
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")   

        return feature_names_sorted, feature_scores, total_time
        
        
    def boruta_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://github.com/scikit-learn-contrib/boruta_py
            Compiled_by: JYang
        Returns: 
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """   
        start_time = time.time()
        # Calcuate boruta feature importance
        feature_selector = BorutaPy(self.model_init, n_estimators='auto', verbose=0, random_state=self.seed)
        feature_selector.fit(self.X_data_val.values, self.y_data_val.values)
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nRuntime: {total_time:.2f} seconds")   
        # Order features by Boruta rank
        selected_features = pd.DataFrame({'feature_name': self.feature_names, 'ranking': feature_selector.ranking_})
        selected_features = selected_features.sort_values(by=['ranking', 'feature_name']) 
        selected_features['score'] = np.arange(len(self.feature_names), 0, -1)
        # Feature names and scores to be returned
        feature_names_sorted = selected_features['feature_name'].tolist()
        feature_scores = selected_features['score'].tolist()
        
        return feature_names_sorted, feature_scores, total_time
        
    def compute_sage_val(self):
        """ A method for computing feature importance using the Sage method
        Returns:
            sage_features (list): ranked features
            values (list): ranked feature importances
        """
        # Calculate sage values
        self.model_init.fit(self.X_data_val.values, self.y_data_val.values)
        imputer = sage.MarginalImputer(self.model_init, self.X_data_train[:512].values)
        estimator = sage.KernelEstimator(imputer, 'cross entropy' if self.pred_type == 'classification' else 'mse', random_state=self.seed)
        self.sage_val = estimator(self.X_data_val.values, self.y_data_val.values, thresh=0.025)
        # Order sage values
        values = self.sage_val.values
        argsort = np.argsort(values)[::-1]
        values = list(values[argsort])
        sage_features = list(np.array(self.feature_names)[argsort])    
        
        return sage_features, values

    def sage_plot(self):
        # Plot sage values
        self.sage_val.plot(self.feature_names)
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=8)
        plt.show()

    def sage_importance(self):
        """ A method that calls the sage class to extract the features, feature scores, and total runtime
            Paper: https://arxiv.org/abs/2004.00668
            Source: https://github.com/iancovert/sage
            Compiled_by: JYang
        Returns:
            sage_features (list): ranked features
            sage_feature_scores (list): ranked feature importances
            total_time (float): total runtime for the Sage feature selection process
        """ 
        start_time = time.time()
        # Generate the Sage Values
        sage_features, sage_feature_scores = self.compute_sage_val()
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: 
            print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
            display(self.sage_plot())

        return sage_features, sage_feature_scores, total_time

    def permutation_test(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://scikit-learn.org/stable/modules/permutation_importance.html
            Compiled_by: JYang
        Returns: 
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time (float): total runtime for the Sage feature selection process
        """      
        start_time = time.time()
        self.model_init.fit(self.X_data_train, self.y_data_train)
        permu_test = permutation_importance(self.model_init, self.X_data_val, self.y_data_val, random_state = self.seed)
        sorted_idx = permu_test.importances_mean.argsort()[::-1]
        feature_names_sorted = [self.feature_names[i] for i in sorted_idx]
        feature_scores = list(permu_test.importances_mean[sorted_idx])
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train:  
            print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
            plt.barh(feature_names_sorted, permu_test.importances_mean[sorted_idx])
            plt.xlabel("Permutation Importance")
        
        return feature_names_sorted, feature_scores, total_time

    def xgb_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://xgboost.readthedocs.io/en/stable/R-package/discoverYourData.html
            Compiled_by: JYang
        Returns:
            top_features (list): ranked features
            top_scores (list): ranked feature importances
            total_time (float): total runtime for the Sage feature selection process
        """     
        # make predictions for val data
        self.model_init.fit(self.X_data_train, self.y_data_train)
        y_pred = self.model_init.predict(self.X_data_val)
        # round predictions
        predictions = [np.round(value) for value in y_pred]
        # evaluate predictions
        #accuracy = accuracy_score(predictions, self.y_data_val)
        start_time = time.time()
        # Calculate feature importance scores
        feature_importances = self.model_init.get_booster().get_score(importance_type='weight')
        
        if len(feature_importances.values()) < 2:
            top_features, top_scores = list(self.X_data_train.columns), [0]*len(self.X_data_train.columns)
        else:
            # Sort the feature importance scores
            sorted_idx = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            # Extract the top feature names and scores
            top_features, top_scores = zip(*sorted_idx[:-1])

        end_time = time.time()
        total_time = end_time - start_time                             
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        
        return top_features, top_scores, total_time

    def shap_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
            Compiled_by: JYang
        Returns:
            sorted_features (list): ranked features
            feature_score_shap (list): ranked feature importances
            total_time (float): total runtime for the Sage feature selection process
        """        
        start_time = time.time()
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X_data_val)
        #shap.plots.waterfall(shap_values[0])
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        # Average the absolute values of the importance scores
        mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        features = np.array(self.feature_names)
        # Sort the features based on their mean absolute SHAP values
        argsort = np.argsort(mean_abs_shap_values)[::-1]
        sorted_features = list(features[argsort])

        if self.print_outputs_train: 
            print("Top features:", sorted_features)
            shap.plots.beeswarm(shap_values, max_display=25)

        feature_score_shap = list(np.mean(shap_values.values, axis=0)[np.argsort(np.mean(shap_values.values, axis=0))[::-1]])            

        return sorted_features, feature_score_shap, total_time                   

    def cae_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Paper: https://arxiv.org/abs/1901.09346
            Source: https://github.com/mfbalin/Concrete-Autoencoders
            Compiled_by: JYang
        Returns:
            sorted_features (list): ranked features
            feature_scores (list): ranked feature importances
            total_time (float): total runtime for the Sage feature selection process
        """     
        (x_train, y_train), (x_val, y_val) = (self.X_data_train.values, self.y_data_train.values), (self.X_data_val.values, self.y_data_val.values)

        x_train = np.reshape(x_train, (len(x_train), -1))
        x_val = np.reshape(x_val, (len(x_val), -1))
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        start_time = time.time()
        
        selector_supervised = ConcreteAutoencoderFeatureSelector(K = len(self.feature_names), rand_seed=self.seed, output_function = g, num_epochs = 3, pred_type=self.pred_type)
        selector_supervised.fit(x_train, y_train, x_val, y_val)

        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        
        feature_importances=selector_supervised.get_support(indices = True)
        argsort = np.argsort(feature_importances)[::-1]
        features = np.array(self.feature_names)
        sorted_features = list(features[argsort])
        feature_scores = list(feature_importances[argsort])

        return sorted_features, feature_scores, total_time
    

    def lasso_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
            Compiled_by: JChang
        Returns:
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """
        start_time = time.time()
        # Calcuate lasso feature importance
        # Create an instance of the Lasso model with the desired alpha value
        lasso = Lasso(alpha=0.00001, random_state=self.seed)
        lasso.fit(self.X_data_train.values, self.y_data_train.values)
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        # Order features by lasso rank
        feature_imp = pd.DataFrame({'Value': np.abs(lasso.coef_), 'Feature': self.feature_names})
        top_features =feature_imp.sort_values(by="Value",ascending=False)
        # Feature names and scores to be returned
        feature_names_sorted=top_features['Feature'].tolist()
        feature_scores=top_features['Value'].tolist()

        return feature_names_sorted, feature_scores, total_time

    def cart_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            Compiled_by: JChang
        Returns:
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """
        start_time = time.time()
        # Calcuate lasso feature importance
        # Create an instance of the Lasso model with the desired alpha value
        cart_model = DecisionTreeClassifier(random_state=self.seed).fit(self.X_data_train.values, self.y_data_train.values) if self.pred_type == "classification" else DecisionTreeRegressor(random_state=self.seed).fit(self.X_data_train.values, self.y_data_train.values)
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        # Order features by CART rank
        feature_imp = pd.DataFrame({'Value': cart_model.feature_importances_, 'Feature': self.feature_names})
        top_features =feature_imp.sort_values(by="Value",ascending=False)
        # Feature names and scores to be returned
        feature_names_sorted=top_features['Feature'].tolist()
        feature_scores=top_features['Value'].tolist()

        return feature_names_sorted, feature_scores, total_time

    def svm_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            Compiled_by: JChang
        Returns:
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """
        start_time = time.time()
        # Calcuate lasso feature importance
        # Create an instance of the Lasso model with the desired alpha value
        clf = SVC(kernel='linear', gamma=0.1, C=1) if self.pred_type == "classification" else SVR(kernel='linear', gamma=0.1, C=1)
        clf.fit(self.X_data_train.values, self.y_data_train.values)
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        # Order features by SVM rank
        feature_imp = pd.DataFrame({'Value': np.abs(clf.coef_[0]), 'Feature': self.feature_names})
        top_features = feature_imp.sort_values(by="Value",ascending=False)
        # Feature names and scores to be returned
        feature_names_sorted=top_features['Feature'].tolist()
        feature_scores=top_features['Value'].tolist()

        return feature_names_sorted, feature_scores, total_time

    def randomforest_importance(self):
        """ A method that extracts the features, feature scores, and total runtime
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            Compiled_by: JChang
        Returns:
            feature_names_sorted (list): ranked features
            feature_scores (list): ranked feature importances
            total_time(float): total runtime for the Sage feature selection process
        """
        start_time = time.time()
        # Calcuate lasso feature importance
        # Create an instance of the Lasso model with the desired alpha value
        rf_model = RandomForestClassifier(n_estimators = 100,random_state=self.seed) if self.pred_type == "classification" else RandomForestRegressor(n_estimators = 100,random_state=self.seed)
        rf_model.fit(self.X_data_train.values, self.y_data_train.values)
        end_time = time.time()
        total_time = end_time - start_time
        if self.print_outputs_train: print(f"\nFeature Selection Runtime: {total_time:.2f} seconds")
        # Order features by SVM rank
        feature_imp = pd.DataFrame({'Value': rf_model.feature_importances_, 'Feature': self.feature_names})
        top_features =feature_imp.sort_values(by="Value",ascending=False)
        # Feature names and scores to be returned
        feature_names_sorted=top_features['Feature'].tolist()
        feature_scores=top_features['Value'].tolist()

        return feature_names_sorted, feature_scores, total_time

# NOTE:
# https://github.com/mfbalin/Concrete-Autoencoders/blob/master/concrete_autoencoder/concrete_autoencoder/__init__.py
# Updated version of the original codes from the author of the Concrete Auto Encoder feature selection method
# Sigmoid function used in place of the softmax function

class ConcreteSelect(Layer):
    def __init__(self, output_dim, rand_seed, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        self.rand_seed = rand_seed

    def build(self, input_shape):
        self.temp = self.add_weight(name='temp', shape=[], initializer=Constant(self.start_temp), trainable=False)
        self.logits = self.add_weight(name='logits', shape=[self.output_dim, input_shape[1]], initializer=glorot_normal(seed=self.rand_seed), trainable=True)
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0, seed=self.rand_seed)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        #noisy_logits = (self.logits) / temp
        #samples = K.softmax(noisy_logits)
        samples = K.sigmoid(noisy_logits)

        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])

        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))

        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class StopperCallback(EarlyStopping):

    def __init__(self, mean_max_target = 0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)

    def on_epoch_begin(self, epoch, logs = None):
        print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))

    def get_monitor_value(self, logs):
        #monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        monitor_value = K.get_value(K.mean(K.max(K.sigmoid(self.model.get_layer('concrete_select').logits))))
        return monitor_value

class ConcreteAutoencoderFeatureSelector():

    def __init__(self, K, output_function, num_epochs = 300, batch_size = None, learning_rate = 0.001, start_temp = 10.0, min_temp = 0.1, tryout_limit = 5, rand_seed=42, pred_type=None):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.rand_seed = rand_seed
        self.pred_type = pred_type

    def fit(self, X, Y = None, val_X = None, val_Y = None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size

        for i in range(self.tryout_limit):

            K.set_learning_phase(1)
            inputs = Input(shape = X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, self.rand_seed, self.start_temp, self.min_temp, alpha, name = 'concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features, self.pred_type)

            self.model = Model(inputs, outputs)

            loss_type = "binary_crossentropy" if self.pred_type == "classification" else "mean_squared_error"
            self.model.compile(Adam(self.learning_rate), loss = loss_type)

            print(self.model.summary())

            stopper_callback = StopperCallback()

            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose = 1, callbacks = [stopper_callback], validation_data = validation_data)#, validation_freq = 10)

            #if K.get_value(K.mean(K.max(K.softmax(self.concrete_select.logits, axis = -1)))) >= stopper_callback.mean_max_target:
            if K.get_value(K.mean(K.max(K.sigmoid(self.concrete_select.logits)))) >= stopper_callback.mean_max_target:
                break

        num_epochs *= 2

        #self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.probabilities = K.get_value(K.sigmoid(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

        return self

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits), self.model.get_layer('concrete_select').logits.shape[1]), axis = 0))

    def transform(self, X):
        return X[self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices = False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model


def g(x, pred_type):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    output_dim = 2 if pred_type == "classification" else 1
    x = Dense(output_dim, activation='sigmoid')(x)
    return x
