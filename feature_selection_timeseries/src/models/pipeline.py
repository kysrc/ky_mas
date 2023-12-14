# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the method(s) that consolidate multiple methods into a wrapped run function to execute the pipeline

import numpy as np
import pandas as pd
from collections import Counter
from feature_selection_timeseries.src.models.predict_model import computeScore 
from feature_selection_timeseries.src.models.utils import create_df, check_column_types 
from feature_selection_timeseries.src.preprocessing.preprocessor import Preprocess
from feature_selection_timeseries.src.models.train_model import generateModel
from feature_selection_timeseries.src.features.feature_selection import featureValues
from feature_selection_timeseries.src.models.test_model import computeTestScore
from feature_selection_timeseries.src.models.utils import rebalance_data, average_cv_outputs, roll_window_split, generate_val_splits 
from feature_selection_timeseries.src.visualization.visualize import plotScore, plot_ts
import matplotlib.pyplot as plt
import time
import joblib
import ast
from tqdm import tqdm


class run:
    """
    Class for running a set of experiments on various combinations of setup.
    
    Attributes:
        cross_validation_type(str): Whether to use moving or expanding window for cross validation
        save_output_file (bool): Whether to save the training and testing outputs.
        raw_df (pd.DataFrame): The raw dataset for training, validation, and testing
        y (pd.DataFrame): Contains the true values.
        train_test_list (list): List of tuples representing train and test split details.
        methods (list): List of methods to evaluate.
        rebalance_type (list): List of rebalance types to consider.
        label_cols (list): List of columns containing labels.
        do_not_encode_cols (list): List of columns not to be encoded.
        seed (int): Seed for reproducibility.
        target_colname (str): Column name for the target variable.
        dataset_name (str): Name of the dataset.
        pred_type (str): Type of prediction.
        append_to_full_df (bool): Whether to append results to the full dataframe.
        n_features (int): Number of features to consider.
        feature_direction (str): Direction of feature selection.
        train_outputs_file_name (str): File name for saving training outputs.
        current_date (str): Current date for timestamping files.
        scaler_filename (str): File name for saving scaler object.
        encoder_filename (str): File name for saving encoder object.
        label_encoder_filename (str): File name for saving label encoder object.
        test_output_file_name (str): File name for the test data outputs
        test_pred_file_name (str): File name for the test data predictions
        print_outputs_train (bool): Whether to display printout for train outputs
        print_outputs_train (bool): Whether to display printout for test outputs
    """
    def __init__(self, cross_validation_type, save_output_file, raw_df, y, train_test_list, methods, rebalance_type, label_cols, 
                  do_not_encode_cols, seed, target_colname, dataset_name, pred_type, append_to_full_df, n_features, 
                  feature_direction, train_outputs_file_name, current_date, scaler_filename, encoder_filename, 
                  label_encoder_filename, test_output_file_name, test_pred_file_name, print_outputs_train, print_outputs_test
                 ):
        self.cross_validation_type = cross_validation_type
        self.save_output_file = save_output_file
        self.raw_df = raw_df
        self.y = y
        self.train_test_list = train_test_list
        self.methods = methods
        self.rebalance_type = rebalance_type
        self.label_cols = label_cols
        self.do_not_encode_cols = do_not_encode_cols
        self.seed = seed
        self.target_colname = target_colname
        self.dataset_name = dataset_name
        self.pred_type = pred_type
        self.append_to_full_df = append_to_full_df
        self.n_features = n_features
        self.feature_direction = feature_direction
        self.train_outputs_file_name = train_outputs_file_name
        self.current_date = current_date
        self.scaler_filename = scaler_filename
        self.encoder_filename = encoder_filename
        self.label_encoder_filename = label_encoder_filename
        self.test_output_file_name = test_output_file_name
        self.test_pred_file_name = test_pred_file_name
        self.print_outputs_train = print_outputs_train
        self.print_outputs_test = print_outputs_test

        self.scaler_saved = None 
        self.encoder_saved = None 
        self.label_encoder_saved = None 
        self.param_feature_metrics_dict = {}
        self.train_ind = None
        self.test_ind = None

        # An empty dataframe to store results
        self.full_df = pd.DataFrame({
            "timestamp": [],
            "feature": [],
            "score": [],
            "feature_reversed": [],
            "score_reversed_rank": [],
            "method": [],
            "dataset": [],
            "dataset_size": [],
            "runtime_sec": [],
            "prediction_type": [],
            "feature_score": [],
            "cm_val": [],
            "cm_val_reversed":[],
            "rebalance": [],
            "rebalance_type": [],
            "data_shape": [],
            "is_best_score": [],
            "cv_iteration": []
        })


    def train(self):
        """
        Performs the training and evaluation process for various combinations
        """
        for n in range(len(self.train_test_list)):
            start_time = time.time()
            for method in tqdm(self.methods, total=len(self.methods), desc="1. Method"):

                self.train_ind, self.test_ind = roll_window_split(num_rolling_windows=self.train_test_list[0][2], dataset_size=np.shape(self.raw_df)[0], test_size=self.train_test_list[0][1])
                
                if self.print_outputs_train:
                    print("\n\n\nroll_window_split:")
                    print("self.train_ind: ", self.train_ind)
                    print("self.test_ind: ", self.test_ind)

                for train_idx, test_idx in tqdm(zip(self.train_ind, self.test_ind), total=len(self.train_ind), desc="2. Backtesting window"):  # Backtesting windows
                    
                    if self.print_outputs_train:
                        print(f"\ntrain_idx: {train_idx}")
                        print(f"self.raw_df.iloc[{str(train_idx[0])}:{str(train_idx[1])}, :]")

                        print(f"\ntest_idx: {test_idx}")
                        print(f"self.raw_df.iloc[{str(test_idx[0])}:{str(test_idx[1])}, :]\n")

                    backtesting_train_window = train_idx
                    backtesting_test_window = test_idx

                    for rt in self.rebalance_type:
                        rebalance = False if rt=="None" else True
                        if self.print_outputs_train: print(f"\n\nMethod: {method}\nDataset: {self.dataset_name}\nRebalance_type: {rt}\nRebalance: {rebalance}\n")
                        param_feature_metrics, self.full_df, self.scaler_saved, self.encoder_saved, self.label_encoder_saved = compile_one_sweep(
                            print_outputs_train = self.print_outputs_train,
                            backtesting_train_window = backtesting_train_window,
                            cross_validation_type=self.cross_validation_type,
                            label_cols=self.label_cols, 
                            do_not_encode_cols=self.do_not_encode_cols, 
                            seed=self.seed,
                            target_colname=self.target_colname,
                            data=self.raw_df.loc[train_idx[0]:train_idx[1]-1, :],
                            full_df=self.full_df,
                            method_name=method,
                            dataset_name=self.dataset_name,
                            pred_type=self.pred_type,
                            num_cv_splits=self.train_test_list[n][2],
                            rebalance=rebalance,
                            rebalance_type=rt,
                            append_to_full_df=self.append_to_full_df,
                            train_examples=self.train_test_list[n][0],
                            test_examples=self.train_test_list[n][1],
                            n_features=self.n_features,
                            feature_direction=self.feature_direction,
                            verbose = True
                        )

                        # Save the best hyperparams and metrics for each n, method, dataset, and rt
                        self.param_feature_metrics_dict[(n, method, self.dataset_name, rt, str(backtesting_train_window), str(backtesting_test_window))] = param_feature_metrics
                        
                        # print(np.shape(self.full_df))
                        # if self.save_output_file: 
                        #     self.full_df.to_excel(f"{self.train_outputs_file_name}{self.current_date}.xlsx")
                        #     print(f"Train outputs saved to: {self.train_outputs_file_name}{self.current_date}.xlsx")

            end_time = time.time()
            total_time = end_time - start_time
            if self.print_outputs_train: print(f"\nTotal Runtime: {total_time:.2f} seconds")

        joblib.dump(self.scaler_saved, self.scaler_filename)
        joblib.dump(self.encoder_saved, self.encoder_filename)
        joblib.dump(self.label_encoder_saved, self.label_encoder_filename)

    def test(self):
        """
        Performs the testing and evaluation process 
        """
        self.scaler_saved = joblib.load(self.scaler_filename)
        self.encoder_saved = joblib.load(self.encoder_filename)
        self.label_encoder_saved = joblib.load(self.label_encoder_filename)

        compiled_metrics = {
            'train_test_list_n': [], 
            'cur_method': [], 
            'cur_bt_train_window': [],
            'cur_bt_test_window': [],
            'rebalance': [], 
            'feature': [], 
            'params_': [], 
            'num_features': [], 
            'test_score': []
          }
        pred_consolidated = pd.DataFrame()

        for k, v in self.param_feature_metrics_dict.items(): # For each train_test_list_n, method, dataset, rebalance combination
            test_input_df = self.param_feature_metrics_dict[k]
            selected_features = list(test_input_df['feature'])
            selected_params = list(test_input_df['params_'])
            selected_num_features = list(test_input_df['num_features'])
                    
            for i in range(len(selected_features)): # Each combination has multiple number of features
                train_test_list_n = k[0]
                cur_method = k[1]
                rebalance = k[3]
                cur_bt_train_window = eval(k[4])
                cur_bt_test_window = eval(k[5])
                cur_features = selected_features[i]
                cur_params = selected_params[i]
                cur_num_features = selected_num_features[i]

                if self.print_outputs_test:
                    print("cur_bt_train_window: ", cur_bt_train_window)
                    print("cur_bt_test_window: ", cur_bt_test_window)
                    print(f"train_data = self.raw_df.loc[ {str(cur_bt_train_window[0])} : {str(cur_bt_train_window[1])} , :]")
                    print(f"test_data = self.raw_df.loc[ {str(cur_bt_test_window[0])} : {str(cur_bt_test_window[1])} , :]")

                computerA = computeTestScore(
                    label_cols=self.label_cols, 
                    do_not_encode_cols=self.do_not_encode_cols,
                    selected_features=cur_features,
                    train_data=self.raw_df.loc[cur_bt_train_window[0]:cur_bt_train_window[1]-1, :].reset_index(drop=True), # Shift back 1 due to loc including endpoints
                    test_data=self.raw_df.loc[cur_bt_test_window[0]:cur_bt_test_window[1]-1, :].reset_index(drop=True),
                    pred_type=self.pred_type,
                    seed=self.seed,
                    print_outputs_test=self.print_outputs_test,
                    scaler_saved=self.scaler_saved,
                    encoder_saved=self.encoder_saved,
                    label_encoder_saved=self.label_encoder_saved
                )
                # metrics: A dict of containing RMSE and num of features, results: a dataframe of true values and preds
                metrics, pred_df = computerA.pred(params=eval(cur_params)) # cur_params was converted to a string in the function compile_one_sweep()
                
                if self.print_outputs_test:
                    # Plot time series
                    plot_ts(pred_df)
                    print("Method: ", cur_method)
                    print("Number of features: ", cur_num_features)
                    print("Best Hyperparams: ", cur_params)
                    print("Score: ", metrics['rmse'], "\n")
                    print("Features: ", cur_features)

                # Store iteration info to dictionary
                compiled_metrics['train_test_list_n'].append(train_test_list_n)
                compiled_metrics['cur_method'].append(cur_method)
                compiled_metrics['cur_bt_train_window'].append(cur_bt_train_window)
                compiled_metrics['cur_bt_test_window'].append(cur_bt_test_window)
                compiled_metrics['rebalance'].append(rebalance)
                compiled_metrics['feature'].append(cur_features)
                compiled_metrics['params_'].append(cur_params)
                compiled_metrics['num_features'].append(cur_num_features)
                compiled_metrics['test_score'].append(metrics['rmse'])

                # Add additional fields to the dataframe with the y_pred and y_true
                len_pred_df = np.shape(pred_df)[0]
                pred_df_addon = pd.DataFrame({
                  'ind': np.arange(0, len_pred_df),
                  'train_test_list_n': [train_test_list_n]*len_pred_df,
                  'cur_method': [cur_method]*len_pred_df,
                  'cur_bt_train_window': [cur_bt_train_window]*len_pred_df,
                  'cur_bt_test_window': [cur_bt_test_window]*len_pred_df,
                  'rebalance': [rebalance]*len_pred_df,
                  'feature': [cur_features]*len_pred_df,
                  'params_': [cur_params]*len_pred_df,
                  'num_features': [cur_num_features]*len_pred_df,
                  'test_score': [metrics['rmse']]*len_pred_df
                  }) 
                # Dataframe containing the original fields in the true label field
                original_y_fields = self.y.loc[cur_bt_test_window[0]:cur_bt_test_window[1]-1, :].reset_index(drop=True)
                # Add other columns
                pred_concat = pd.concat([original_y_fields, pred_df.loc[:, "y_pred"], pred_df_addon], axis=1)
                # Row bind pred_consolidated from other iterations
                pred_consolidated = pd.concat([pred_consolidated, pred_concat], axis=0)

        compiled_metrics_df = pd.DataFrame(compiled_metrics)

        compiled_metrics_df_avg = compiled_metrics_df.groupby(['cur_method', 'num_features', 'rebalance', 'train_test_list_n'])['test_score'].mean().reset_index()
       
        print("\nAverage test score (RMSE) of each method for all backtesting windows: \n")
        display(compiled_metrics_df_avg)
        print("\n")

        # Export test data results
        if self.save_output_file:
            # Save test results
            compiled_metrics_df.to_excel(f"{self.test_output_file_name}{self.current_date}.xlsx", index=False)
            if self.print_outputs_test: print(f"Test outputs saved to: {self.test_output_file_name}{self.current_date}.xlsx")
            # Save predictions
            pred_consolidated.to_excel(f"{self.test_pred_file_name}{self.current_date}.xlsx", index=False)
            if self.print_outputs_test: print(f"Test preds saved to: {self.test_pred_file_name}{self.current_date}.xlsx")


def compile_one_sweep(print_outputs_train, backtesting_train_window, cross_validation_type, label_cols, do_not_encode_cols, seed, target_colname, data, full_df, method_name, 
                      dataset_name, pred_type, num_cv_splits=5, rebalance=False, rebalance_type=None, append_to_full_df=False, 
                      train_examples=1, test_examples=1, n_features=0, feature_direction="both", verbose=False):
    """ A method that runs through 1 sweep of the entire pipeline by wrapping the required methods
    Args:
        print_outputs_train (bool): whether to print train outputs
        backtesting_train_window (list): a list of lists containing training start and end indices
        cross_validation_type (str): whether to use moving or expanding window
        label_cols (list): list of columns to label encode
        do_not_encode_cols (list): list of columns to not encode
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
        data (dict): a dictionary containing train and validation data
        full_df (dataframe): a dataframe containing all currently tracked model results
        method_name (str): name of the feature selection model
        dataset_name (str): name of the dataset
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        num_cv_splits (int): an integer indicating the number of cross validation splits
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        append_to_full_df (bool): a boolean indicating whether to append model results to the existing tracked results
        train_examples (int): number of train examples in each cv split
        test_examples (int): number of test examples in each cv split
        n_features (int): number of features to use; used for top n and bottom n features
        feature_direction (str): a string indicating whether to evaluate top/bottom/both ranking features
    Returns:      
        full_df (dataframe): a dataframe containing all currently tracked model results
        scaler_saved (object): an object for rescaling numerical features
        encoder_saved (object): an object for encoding categorical features
        label_encoder_saved (obj): an object for label encoding
    """   
    # Extract categorical and numerical features
    categorical_cols, numerical_cols = check_column_types(data.iloc[:,:-1], label_cols, do_not_encode_cols)
    if print_outputs_train:
        print("Categorical Columns: ", categorical_cols)
        print("Numerical Columns: ", numerical_cols)
        print("Label Encode Columns: ", label_cols)
        print("Do Not Encode Columns: ", do_not_encode_cols)
        
    # Preprocessing the data
    processor1 = Preprocess(
        print_outputs_train = print_outputs_train, 
        backtesting_train_window = backtesting_train_window,
        data = data,
        target = target_colname,
        cat_cols = categorical_cols,
        num_cols = numerical_cols, 
        label_cols = label_cols,  
        do_not_encode_cols = do_not_encode_cols, 
        num_cv_splits = num_cv_splits,
        train_examples = train_examples,
        test_examples = test_examples,
        cross_validation_type=cross_validation_type
    )
    # Dictionary containing all cross validation splits
    compiled_data, scaler_saved, encoder_saved, label_encoder_saved, train_test_index = processor1.split_data()
    # The number of cv splits
    cv_iteration = len(compiled_data['X_train'])
    
    model_generator = generateModel(pred_type=pred_type, seed=seed)
    # Intialize hyerparams
    params, param_combinations = model_generator.generate_hyperparam_combo()
    # Keep track of scoring metrics
    score_tracker = {'cv': [], 'params': [], 'num_features': [], 'score': [], 'y_pred': [], 'feature_scores': [], 'features': []}

    for grid_params in tqdm(param_combinations, total=len(param_combinations), desc="3. Hyperparam combinations"):
        grid_params['max_depth'] = int(grid_params['max_depth'])
        params.update(grid_params)

        # Loop for all cv splits and compute scoring metrics
        for i in tqdm(range(cv_iteration), total=cv_iteration, desc="4. train-validation splits"):
            selected_cv_dict = {}
            # Iterate through the original dictionary
            for key, df_list in compiled_data.items():
                selected_cv_dict[key] = df_list[i]

            trained_model = model_generator.get_model(data_dict=selected_cv_dict, params=params)

            if print_outputs_train:
                #display(selected_cv_dict)
                print("\n\n\n---------------------------------------------------------------------------------------------------")
                print('\nX_train', np.shape(selected_cv_dict['X_train']))
                print('X_val', np.shape(selected_cv_dict['X_val']))
                print('y_train', np.shape(selected_cv_dict['y_train']))
                print('y_val', np.shape(selected_cv_dict['y_val']), "\n")
                print(f"Running Cross-Validation Split: train_index=[{train_test_index['train_index'][i][0]}, {train_test_index['train_index'][i][-1]}], test_index=[{train_test_index['test_index'][i][0]}, {train_test_index['test_index'][i][-1]}]\n")

            # Compute metrics
            full_df, score_tracker = get_metrics_df(
                print_outputs_train = print_outputs_train,
                cv = i,
                score_tracker=score_tracker,
                params = grid_params,
                trained_model=trained_model,
                seed=seed, 
                target_colname=target_colname, 
                data_original=data,
                data=selected_cv_dict, 
                full_df=full_df, 
                method_name=method_name, 
                dataset_name=dataset_name, 
                pred_type=pred_type, 
                cv_iteration=i,
                train_examples=train_examples,
                test_examples=test_examples,
                num_cv_splits=num_cv_splits,
                rebalance=rebalance, 
                rebalance_type=rebalance_type, 
                append_to_full_df=append_to_full_df,
                n_features=n_features,
                feature_direction=feature_direction
            )

    # Converted tracked metrics into a dataframe
    track_metrics_df = pd.DataFrame(score_tracker)
    track_metrics_df['params_'] = track_metrics_df['params'].astype(str)
    # Average the cv scores for each params and num_features combination
    track_metrics_df_grouped = track_metrics_df.groupby(['params_', 'num_features'])['score'].mean().reset_index()

    if pred_type == "classification":
        result_df = track_metrics_df_grouped.groupby(['num_features'])['score'].idxmax()
    else:
        result_df = track_metrics_df_grouped.groupby(['num_features'])['score'].idxmin()
    # The resulting dataframe for the best param choices
    desired_hyperparam = track_metrics_df_grouped.loc[result_df]
    # Average the feature scores generalized for all cv splits (features sorted by average feature scores)
    average_feature_metrics = average_cv_outputs(track_metrics_df, cv_iteration)
    # Merge the best param choices with the generalized features
    param_feature_metrics = pd.merge(desired_hyperparam, average_feature_metrics, on=['params_', 'num_features'], how='inner')

    if print_outputs_train:
        print("\ntrack_metrics_df:") 
        display(track_metrics_df)
        print("\ntrack_metrics_df_grouped:") 
        display(track_metrics_df_grouped)
        print("\ndesired_hyperparam:")
        display(desired_hyperparam)
        print("\naverage_feature_metrics:")
        display(average_feature_metrics)
        print("\nparam_feature_metrics:")
        display(param_feature_metrics)

    return param_feature_metrics, full_df, scaler_saved, encoder_saved, label_encoder_saved


def run_scoring_pipeline(print_outputs_train, cv, score_tracker, params, feature_impt, feature_scores, n_features, input_data_dict, pred_type, rebalance, rebalance_type, seed, feature_direction):
    """ A method for generating model prediction scoring metrics
    Args:
        print_outputs_train (bool): whether to print train outputs
        cv (int): cross-validation split 
        score_tracker (dict): a dictionary containing info for tracking scoring metrics
        params (dict): a dictionary of hyperparams
        feature_impt (list): list of top features to use for model prediction
        feature_scores (list): a list of feature scores
        n_features (int): number of features to use; used for top n and bottom n features
        input_data_dict (dict): a dictionary containing train and validation data
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        seed (int): a random state
        feature_direction (str): a string indicating whether to evaluate top ranking features or both top and bottom ranking features: top or both
    Returns:    
        all_scores (list): a list containing the prediction score for the top n features
        all_scores_reverse (list): a list containing the prediction score for the bottom n features
        all_features (list): a list of top n features subsets
        all_features_reverse (list): a list of bottom n features subsets
        all_cm_val (list): a list of other scoring metrics for top n features
        all_cm_val_reverse (list): a list of other scoring metrics for bottom n features
    """
    # Rebalance the dataset
    if rebalance:
        input_data_dict_rebalanced = rebalance_data(data=input_data_dict, rebalance_type=rebalance_type, seed=seed)
   
    all_scores = []
    all_features = []
    all_cm_val = []
    all_scores_reverse = []
    all_features_reverse = []
    all_cm_val_reverse = []
    
    # A list indicating the number of features to run the test for
    if isinstance(n_features, int):
        # If the provided value is an integer, run the test for the number of features equal to that value and the total number of features
        use_num_features = sorted([int(i) for i in set([min(n_features, len(feature_impt)), len(feature_impt)])])
    else:
        # If the provided value is a string, run the test for all combinations of features from 1 to the entire feature set
        use_num_features = list(np.arange(1, len(feature_impt)+1))
    
    # For top n and all features, compute and save the scoring metrics
    if feature_direction in ['top', 'both']:
        for i in use_num_features:      
            if print_outputs_train: print(f"Number of features: {i}")  
            # Instantiate an object for computing scoring metrics
            compute_score_1 = computeScore(
                data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
                keep_cols=feature_impt[:i] if i < len(feature_impt) else list(input_data_dict["X_train"].keys()),
                pred_type=pred_type,
                seed=seed,
                params=params
              )

            if pred_type == "regression":
                score, y_pred, cm_val, df_for_ts = compute_score_1.pred_score() 
                if print_outputs_train: 
                    plot_ts(df_for_ts)
                    #print("predictions:")
                    #print(list(y_pred))
                    #print("true_values:")
                    #print(list(df_for_ts["y_true"]), "\n")
            else:
                score, y_pred, cm_val, _ = compute_score_1.pred_score() 
            
            all_scores.append(score) # score is a scalar
            all_features.append(feature_impt[:i])
            all_cm_val.append(cm_val)
            score_tracker['cv'].append(cv)
            score_tracker['params'].append(params)
            score_tracker['num_features'].append(i)
            score_tracker['score'].append(score)
            score_tracker['y_pred'].append(y_pred) 
            score_tracker['features'].append(feature_impt)  
            score_tracker['feature_scores'].append(feature_scores) 
            
            if print_outputs_train: 
                print("params: ", params)   
                print("cv: ", cv) 
                print("num_features: ", i) 
                print("score: ", score, "\n") 
    
    # For each top n and all features in reversed order, compute and save the scoring metrics
    if feature_direction in ['both']:
        for i in use_num_features:     
            if print_outputs_train: print(f"Number of features: {i}")
            compute_score_1 = computeScore(
                data_dict=input_data_dict_rebalanced if rebalance else input_data_dict,
                keep_cols=feature_impt[::-1][:i] if i < len(feature_impt) else list(input_data_dict["X_train"].keys()),
                pred_type=pred_type,
                seed=seed,
                params=params
            )

            if pred_type == "regression":
                score, y_pred, cm_val, df_for_ts = compute_score_1.pred_score() 
                if print_outputs_train: 
                    plot_ts(df_for_ts)
                    #print("predictions:")
                    #print(list(y_pred))
                    #print("true_values:")
                    #print(list(df_for_ts["y_true"]), "\n")
            else:
                score, y_pred, cm_val, _ = compute_score_1.pred_score() 

            all_scores_reverse.append(score)
            all_features_reverse.append(feature_impt[::-1][:i])
            all_cm_val_reverse.append(cm_val) 
       
    if print_outputs_train: 
        # Plot the scores
        if feature_direction in ['top', 'both']:
            print("\n\n-----------------------------------Evaluate Features From Highest Importance-----------------------------------\n")
            plt.close()
            plotter = plotScore(data=all_scores, feature_impt=feature_impt, pred_type=pred_type, use_num_features=use_num_features)
            display(plotter.score_plot())
        
        # Plot the scores for the reversed feature order
        if feature_direction in ['both']: 
            print("\n\n-----------------------------------Evaluate Features From Lowest Importance-----------------------------------\n")
            plt.close()
            plotter_reversed = plotScore(data=all_scores_reverse, feature_impt=feature_impt[::-1], pred_type=pred_type, use_num_features=use_num_features)
            display(plotter_reversed.score_plot())

    return all_scores, all_scores_reverse, all_features, all_features_reverse, all_cm_val, all_cm_val_reverse, score_tracker


def get_metrics_df(print_outputs_train, cv, score_tracker, params, trained_model, seed, target_colname, data, data_original, full_df, method_name, dataset_name, pred_type, cv_iteration, train_examples,
                   test_examples, num_cv_splits, rebalance=False, rebalance_type=None, append_to_full_df=False, n_features=None, feature_direction=None):
    """ A methold for generating the model training results
    Args:
        print_outputs_train (bool): whether to print train outputs
        cv (int): cross-validation split 
        score_tracker (dict): a dictionary containing info for tracking scoring metrics
        params (dict): initialized hyperparams 
        trained_model (obj): a trained model
        seed (int): a random state
        target_colname (str): a string indicating the name of the target variable column
        data (dict): a dictionary containing train (rebalanced, if applicable) and validation data
        data_original (dict): a dictionary containing train and validation data
        full_df (dataframe): a dataframe containing currently tracked model results
        method_name (str): name of the feature selection model
        dataset_name (str): name of the dataset
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        cv_iteration (int): an integer indicating the cross validation split iteration
        train_examples (int): number of train examples in each cv split
        test_examples (int): number of test examples in each cv split
        num_cv_splits (int): an integer indicating the number of cross validation splits
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        append_to_full_df (bool): a boolean indicating whether to append model results to the existing tracking table
        n_features (int): number of features to use; used for top n and bottom n features
        feature_direction (str): a string indicating whether to evaluate top/bottom/both ranking features
    Returns:    
        full_df (dataframe): a dataframe containing all currently tracked model results
    """   
    feature_values = featureValues(print_outputs_train=print_outputs_train, params=params, data_dict=data, pred_type=pred_type, model=trained_model, seed=seed, target_colname=target_colname, n_features=n_features)
        
    # Generate feature ranking and other input variables
    if method_name.lower() == "permutation":
        sorted_features, feature_scores, total_time = feature_values.permutation_test()

    if method_name.lower() == "xgboost":
        sorted_features_xgb, feature_scores, total_time = feature_values.xgb_importance()
        # For missing features, add them to the list of features and assign them with a feature score of 0
        sorted_features = list(sorted_features_xgb) + [f for f in list(data['X_train'].columns) if f not in sorted_features_xgb]
        feature_scores = list(feature_scores) + [0]*(len(data['X_train'].columns) - len(feature_scores))

    if method_name.lower() == "shap":
        sorted_features, feature_scores, total_time = feature_values.shap_importance()

    if method_name.lower() == "boruta":
        sorted_features, feature_scores, total_time = feature_values.boruta_importance()
        
    if method_name.lower() == "sage":
        sorted_features, feature_scores, total_time = feature_values.sage_importance()
        
    if method_name.lower() == "cae":
        sorted_features, feature_scores, total_time = feature_values.cae_importance()
    
    if method_name.lower() == "dynamic":
        sorted_features, feature_scores, total_time = feature_values.dynamic_selection_importance()
        
    if method_name.lower() == "stg":
        sorted_features, feature_scores, total_time = feature_values.stg_importance()
    
    if method_name.lower() == "lasso":
        sorted_features, feature_scores, total_time = feature_values.lasso_importance()
    
    if method_name.lower() == "cart":
        sorted_features, feature_scores, total_time = feature_values.cart_importance()
        
    if method_name.lower() == "svm":
        sorted_features, feature_scores, total_time = feature_values.svm_importance()
        
    if method_name.lower() == "rf":
        sorted_features, feature_scores, total_time = feature_values.randomforest_importance()
        
    # Generate the scoring metrics
    all_scores, all_scores_reverse, all_features, all_features_reverse, cm_val, cm_val_reversed, score_tracker = run_scoring_pipeline(
        print_outputs_train = print_outputs_train,
        cv = cv,
        score_tracker = score_tracker,
        params = params,
        feature_impt = sorted_features,
        feature_scores = feature_scores,
        n_features = n_features,
        input_data_dict = data,
        pred_type = pred_type,
        rebalance=rebalance,
        rebalance_type=rebalance_type,
        seed=seed,
        feature_direction=feature_direction
    )

    if pred_type == "classification":
        y_train_dict = dict(sorted(Counter(data["y_train"]).items()))
        y_val_dict = dict(sorted(Counter(data["y_val"]).items()))
        y_classes_dict = {0: y_train_dict[0] + y_val_dict[0], 1: y_train_dict[1] + y_val_dict[1]}
        best_score = [num_index == all_scores.index(max(all_scores)) for num_index in range(len(all_scores))]
    else:
        y_train_dict = {}
        y_val_dict = {}
        y_classes_dict = {}
        best_score = [num_index == all_scores.index(min(all_scores)) for num_index in range(len(all_scores))]
    
    X_train_shape = np.shape(data["X_train"])
    X_val_shape = np.shape(data["X_val"])

    # Compile dataframe containing scoring metrics for all feature subsets
    results_df = create_df(
        all_features = all_features,
        all_scores = all_scores,
        all_features_rev = all_features_reverse,
        all_scores_rev = all_scores_reverse,
        dataset_size = str(np.shape(data_original)),
        total_time = total_time,
        method_name = method_name,
        dataset_name = dataset_name,
        pred_type = pred_type,
        feature_score = [feature_scores[:i] for i in [len(e) for e in all_features]],  # Keep the number of feature scores equal to the number of features
        cm_val = cm_val,
        cm_val_reversed = cm_val_reversed,
        rebalance = rebalance,
        rebalance_type = rebalance_type,
        data_shape = str({
            "cv_train_size": train_examples,
            "cv_test_size": test_examples,
            "num_cv_split": num_cv_splits,
            "X_train (instance/feature)": X_train_shape,
            "X_val (instance/feature)": X_val_shape,
            "y_train (class/count)": y_train_dict,
            "y_val (class/count)": y_val_dict,
            "X_total (instance/features)": (X_train_shape[0] + X_val_shape[0], X_train_shape[1]),
            "y_total (class/count)": y_classes_dict            # If KeyError => data has a missing class; not enough data 
        }),
        is_best_score = best_score,
        cv_iteration = cv_iteration
    )

    if print_outputs_train: display(results_df.head())
    
    if append_to_full_df:
        full_df = pd.concat([results_df, full_df])
        
    return full_df, score_tracker








