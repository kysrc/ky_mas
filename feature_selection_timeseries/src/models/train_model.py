# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the method(s) for generating the trained XGBoost model

import xgboost as xgb
import torch
import numpy as np
from feature_selection_timeseries.src.models.utils import setup_seed

class generateModel:
    """A class with a method for generating a trained model that includes hyperparameter tuning for the validation and test data
    Args:
        data_dict (dict): a dictionary containing dataframes of the train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        seed (int): a random state
    """
    def __init__(self, pred_type, seed):
        self.pred_type = pred_type.lower()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_seed(self.seed)

    def generate_hyperparam_combo(self):

        params = {
            "objective": "binary:logistic" if self.pred_type == 'classification' else "reg:squarederror",
            "eval_metric": "error" if self.pred_type == 'classification' else "rmse",
            "tree_method": "hist",
            "device": self.device,
            "seed": self.seed
        }

        # param_grid = {
        #     #'n_estimators': [100, 1000, 10000], # Gives warning that it's not in use
        #     'min_child_weight': [0.2, 1, 3], # [default=1]      # Large values lead to underfit
        #     'gamma': [0, 0.001, 0.01],  # [default=0, alias: min_split_loss]   # Large values lead to underfit
        #     #'subsample': [0.6, 1.0], # [default=1]
        #     #'colsample_bytree': [0.4, 0.8, 1.0], #  [default=1]
        #     'max_depth': [6, 25, 100], # [default=6]
        #     'learning_rate': [0.03, 0.3], # [default=0.3, alias: learning_rate]
        #     'alpha': [0, 1, 5]  #  [default=0, alias: reg_alpha]
        # }

        param_grid = {
            'min_child_weight': [0.5, 1], # [default=1]      # Large values lead to underfit
            'gamma': [0, 0.01],  # [default=0, alias: min_split_loss]   # Large values lead to underfit
            'max_depth': [6, 10], # [default=6]
            'learning_rate': [0.03, 0.3], # [default=0.3, alias: learning_rate]
        }
        
        param_combinations = [{param_name: value for param_name, value in zip(param_grid.keys(), combination)}
                              for combination in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))]

        return params, param_combinations

    def get_model(self, data_dict, params, data_type="train"):
        """Train an XGBoost model and return it
        Returns: 
            model (obj): A trained XGBoost model
        """
        X, y = np.array(data_dict["X_train"]), np.array(data_dict["y_train"])
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(data_dict["X_train"].columns))
        # Create a DMatrix for XGBoost training using the best hyperparameters
        if data_type == "train":
            X_val, y_val = np.array(data_dict["X_val"]), np.array(data_dict["y_val"])
            dval  = xgb.DMatrix(X_val, label=y_val, feature_names=list(data_dict["X_train"].columns))
            evals = [(dtrain, 'train'), (dval, 'eval')]
            # Stop training when there's no improvement after 10 rounds
            early_stop = xgb.callback.EarlyStopping(rounds=10, metric_name='rmse', data_name='eval')

            model = xgb.train(params, dtrain, num_boost_round=2000, evals=evals, callbacks=[early_stop], verbose_eval=False)

        else:
            model = xgb.train(params, dtrain, num_boost_round=2000)

        return model

