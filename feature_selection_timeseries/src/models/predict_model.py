# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the method(s) for computing evaluation metrics

import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score
from feature_selection_timeseries.src.models.train_model import generateModel
import torch
import pandas as pd

class computeScore:
    """
    A class for computing evaluation scores based on predictions.

    Args:
        data_dict (dict): A dictionary containing dataframes of the train and validation data.
        keep_cols (list): A list of columns to filter for.
        pred_type (str): A string indicating the type of prediction problem: classification or regression.
        seed (int): A random state.
        params (dict): Model hyperparameters.
    Methods:
        filter_data(): Filters dataframe columns to retain only the specified list of selected features.
        pred_score(): Applies feature filter and generates prediction scores based on the specified prediction type.
    """
    def __init__(self, data_dict, keep_cols, pred_type, seed, params):
        self.data_dict = data_dict
        self.keep_cols = keep_cols 
        self.data_dict_new = {}
        self.pred_type = pred_type.lower()
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params

    def filter_data(self):
        """A method for filtering dataframe columns to retain only the specified list of selected features"""
        # Extract features and labels
        features = [k for k in self.data_dict.keys() if "X_" in k]
        labels = [k for k in self.data_dict.keys() if "y_" in k]
        # Filter features
        for f, l in zip(features, labels):
            if self.data_dict[f] is None:
                self.data_dict_new[f] = []
                self.data_dict_new[l] = []
            else:
                self.data_dict_new[f] = self.data_dict[f][self.keep_cols]
                self.data_dict_new[l] = self.data_dict[l]

    def pred_score(self):
        """
        Generates prediction scores based on the specified prediction type.
        Returns:
            tuple: A tuple containing the prediction score, predicted values, confusion matrix details, and time series dataframe.
        """
        
        # Apply feature filter
        self.filter_data()
        # True labels for the validation set
        y_val = self.data_dict_new['y_val']
        # Generate predictions
        dval = xgb.DMatrix(np.array(self.data_dict_new['X_val']), feature_names=list(self.data_dict_new['X_val'].columns))

        self.trained_model = generateModel(pred_type=self.pred_type, seed=self.seed).get_model(data_dict=self.data_dict_new, params=self.params)

        # Make predictions on the validation set
        if self.pred_type == 'classification':
            y_pred = (self.trained_model.predict(dval) > 0.5).astype(int)
            score = f1_score(y_val, y_pred) # F1 Score

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            cm = confusion_matrix(y_val, y_pred)
            cm_val = {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "total_positive": np.sum(self.data_dict_new['y_val'] == 1),
                "total_negative": np.sum(self.data_dict_new['y_val'] == 0),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred),
                "f1_score": f1_score(y_val, y_pred),
                "accuracy": accuracy_score(y_val, y_pred),
                "num_features": len(self.data_dict_new['X_val'].columns)
                }
            return  score, y_pred, str(cm_val), None
        else:
            y_pred = self.trained_model.predict(dval)

            cm_val = {"num_features": len(self.data_dict_new['X_val'].columns)}
            df_for_ts = pd.DataFrame({ "date": np.arange(0, len(y_pred)),"y_true": y_val.values,  "y_pred": y_pred})
            score = np.sqrt(mean_squared_error(y_val, y_pred))  # RMSE Score

            return  score, y_pred, str(cm_val), df_for_ts

 



