# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the method(s) for evaluating model performance on the test data

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score
from feature_selection_timeseries.src.models.train_model import generateModel
from feature_selection_timeseries.src.models.utils import check_column_types
from feature_selection_timeseries.src.preprocessing.preprocessor import Preprocess

class computeTestScore(Preprocess):
    """ A class for computing the model scores based on a given set of features on the test set
    Args:
        label_cols (list): a list of columns to label encode
        do_not_encode_cols (list): a list of columns to not apply any transformation
        selected_features (list): a list of selected features
        train_data (dataframe): the training data
        test_data (dataframe): the test data
        pred_type (str): type of prediction task ("classification" or "regression")
        seed (int): random seed for reproducibility
        scaler_saved (object): an object for rescaling numerical features
        encoder_saved (object): an object for encoding categorical features
        label_encoder_saved (object): an object for label encoding
        printout (bool, optional): whether to print intermediate outputs. Default is False.
    """
    def __init__(self, label_cols, do_not_encode_cols, selected_features, train_data, test_data, pred_type, seed, scaler_saved, encoder_saved, label_encoder_saved, print_outputs_test):
        self.label_cols = label_cols 
        self.do_not_encode_cols = do_not_encode_cols
        self.selected_features = selected_features
        self.X_train_data = train_data.iloc[:, :-1]
        self.X_test_data = test_data.iloc[:, :-1]
        self.y_train_data = train_data.iloc[:, -1]
        self.y_test_data = test_data.iloc[:, -1]
        self.pred_type = pred_type
        self.seed = seed     
        self.scaler = scaler_saved
        self.encoder = encoder_saved
        self.label_encoder = label_encoder_saved
        self.print_outputs_test = print_outputs_test
        self.cat_cols, self.num_cols = check_column_types(self.X_train_data, self.label_cols, self.do_not_encode_cols)    
        
    def encode_norm(self, X_train, X_test):
        """ A method for data transformation
        Args:
            X_train (dataframe): a dataframe containing train data
            X_test (dataframe): a dataframe containing test data
        Returns:
             X_train_transformed (dataframe): a dataframe containing transformed train data
             X_test_transformed (dataframe): a dataframe containing transformed test data
        """
        #if self.print_outputs_test:
        #    print("X_train")
        #    display(X_train.head())
        #    print("X_test")
        #    display(X_test.head())    
            
        X_train_data_transformed = pd.DataFrame()
        X_test_data_transformed = pd.DataFrame()

        # Normalize numerical variables
        if len(self.num_cols) > 0:
            X_train_scaled = self.scaler.transform(X_train[self.num_cols])
            X_test_scaled = self.scaler.transform(X_test[self.num_cols])
            num_feature_names = [str(f) for f in self.scaler.get_feature_names_out().tolist()]
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train_scaled, columns=num_feature_names)], axis=1)
            X_test_data_transformed = pd.concat([X_test_data_transformed, pd.DataFrame(X_test_scaled, columns=num_feature_names)], axis=1)
            
            #if self.print_outputs_test:
                #print("Added Transformed Numerical Features")
                #display(X_train_data_transformed.head())

        # Encode categorical variables
        if len(self.cat_cols) > 0:
            X_train_encoded = self.encoder.transform(X_train[self.cat_cols]).toarray()
            X_test_encoded = self.encoder.transform(X_test[self.cat_cols]).toarray()
            cat_feature_names = [str(f) for f in self.encoder.get_feature_names_out().tolist()]
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train_encoded, columns=cat_feature_names)], axis=1)
            X_test_data_transformed = pd.concat([X_test_data_transformed, pd.DataFrame(X_test_encoded, columns=cat_feature_names)], axis=1)

            #if self.print_outputs_test:
                #print("Added Transformed Categorical Features")
                #display(X_train_data_transformed.head())

        # Label Encode variables
        if len(self.label_cols) > 0:
            X_train_label_encoded = self.label_encoder.transform(X_train[self.label_cols])
            X_test_label_encoded = self.label_encoder.transform(X_test[self.label_cols])
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train_label_encoded, columns=self.label_cols)], axis=1)
            X_test_data_transformed = pd.concat([X_test_data_transformed, pd.DataFrame(X_test_label_encoded, columns=self.label_cols)], axis=1)

            #if self.print_outputs_test:
                #print("Added Transformed Label Features")
                #display(X_train_data_transformed.head())

        # Features that do not require transformation
        if len(self.do_not_encode_cols) > 0:
            X_train_data_transformed = pd.concat([X_train_data_transformed, pd.DataFrame(X_train[self.do_not_encode_cols].values, columns=self.do_not_encode_cols)], axis=1)
            X_test_data_transformed = pd.concat([X_test_data_transformed, pd.DataFrame(X_test[self.do_not_encode_cols].values, columns=self.do_not_encode_cols)], axis=1)

            #if self.print_outputs_test:
                #print("Added Non-Transformed Features")
                #display(X_train_data_transformed.head())

        return X_train_data_transformed, X_test_data_transformed
    
    def filter_data(self, X_train_transformed, X_test_transformed):
        """ A method for filtering dataframes based on a selected list of features
        Returns:
            X_train_transformed (dataframe): train data with filtered features only
            X_test_transformed (dataframe): test data with filtered features only
        """
        # Dataframes retaining only the selected features
        X_train_filtered = X_train_transformed[self.selected_features]
        X_test_filtered = X_test_transformed[self.selected_features] 
        
        if self.print_outputs_test:
            print("Selected Features:\n", self.selected_features, "\n")
            print("X_train_transformed columns:\n", list(X_train_transformed.columns), "\n")
            print("X_test_transformed columns:\n", list(X_test_transformed.columns), "\n")
            print("Selected features not in X_train_transformed columns:\n", [item for item in self.selected_features if item not in list(X_train_transformed.columns)], "\n")
            print("Selected features not in X_test_transformed columns:\n", [item for item in self.selected_features if item not in list(X_test_transformed.columns)], "\n")

        return X_train_filtered, X_test_filtered
        
    def pred(self, params):
        """ A method for generating prediction metrics on the test data
        Args:
            params (dict): a dictionary containing best the hyperparams
        Returns:
            metrics (dict): a dictionary containing scoring metrics 
        """
        # Encode/Normalize train and test data
        self.X_train_data, self.X_test_data = self.encode_norm(X_train=self.X_train_data, X_test=self.X_test_data)        

        # Filter the dataframes to retain only the specified selected features
        self.X_train_data, self.X_test_data = self.filter_data(X_train_transformed=self.X_train_data, X_test_transformed=self.X_test_data)
        
        # Consolidate the train and test data into a dictionary
        data_dict = {
            "X_train": self.X_train_data,
            "y_train": self.y_train_data,
            "X_test": self.X_test_data,
            "y_test": self.y_test_data
        }
        # Build trained model (with hyperparam tuning)
        trained_model = generateModel(pred_type=self.pred_type, seed=self.seed).get_model(data_dict=data_dict, params=params, data_type="test")
        # Convert dataframe to DMatrix for prediction
        dtest = xgb.DMatrix(np.array(data_dict['X_test']), feature_names=list(data_dict['X_test'].columns))
        # Predictions
        y_pred = trained_model.predict(dtest)
        y_true = data_dict['y_test']

        #if self.print_outputs_test:
        #    print("predictions:")
        #    print(list(y_pred))
        #    print("true_values:")
        #    print(list(y_true), "\n")

        # Compile evaluation metrics
        if self.pred_type == "classification":
            y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
            score = f1_score(y_true, y_pred) #accuracy_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cm = confusion_matrix(y_true, y_pred)
            metrics = {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "total_positive": np.sum(data_dict['y_test'] == 1),
                "total_negative": np.sum(data_dict['y_test'] == 0),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "accuracy": score,
                "num_features": len(data_dict["X_test"].columns)
            }
        else:
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
            }

        return metrics, pd.DataFrame({"y_true": list(y_true), "y_pred": list(y_pred)})



