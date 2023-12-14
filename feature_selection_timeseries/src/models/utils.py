# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the helper method(s), such as tracking tables for model benchmark, data rebalancing, etc.

import pandas as pd
import numpy as np
import datetime
from collections import Counter
import copy
import csv
import openpyxl
import os
import torch
import random
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SMOTENC, BorderlineSMOTE, ADASYN


def roll_window_split(num_rolling_windows, dataset_size, test_size):
    """
    Generates rolling window splits for training and testing indices.
    Args:
        num_rolling_windows (int): Number of rolling windows to generate.
        dataset_size (int): Total size of the dataset.
        test_size (int): Size of the test set.
    Returns:
        tuple: Two lists containing training and testing indices for each rolling window.
    """
    rollig_windows_train = []
    rolling_windows_test = []

    for i in range(num_rolling_windows):
        if len(rollig_windows_train) == 0:
            # For the first window, set initial indices
            train_start = 0
            train_end = test_start = dataset_size - num_rolling_windows*test_size  
            test_end = train_end + test_size
            rollig_windows_train.append([train_start, train_end])
            rolling_windows_test.append([train_end, test_end])
        else:
            # For subsequent windows, update indices based on the previous window
            train_start = rollig_windows_train[-1][0] + test_size 
            train_end = test_start = rolling_windows_test[-1][1] 
            test_end = rolling_windows_test[-1][1] + test_size
            rollig_windows_train.append([train_start, train_end])
            rolling_windows_test.append([test_start, test_end])

        #print(f"train_val_start: {train_start}, train_val_end: {train_end}, test_start: {test_start}, test_end: {test_end}\n")

    return rollig_windows_train, rolling_windows_test

def generate_val_splits(train_start, train_end, test_size, num_splits):
    """
    Generates training and validation indices for a specified number of splits.
    Args:
        train_start (int): Starting index of the training set.
        train_end (int): Ending index of the training set.
        test_size (int): Size of the test set.
        num_splits (int): Number of splits.
    Returns:
        tuple: Two lists containing training and validation indices for each split.
    """
    train_indices = []
    val_indices = []

    train_size = (train_end - train_start - test_size) // num_splits

    for i in range(num_splits):
        if i == 0: # Add remainder to the starting train index only
            remainder = (train_end - train_start - test_size) % num_splits
            train_start += remainder
            current_train_start = train_start + i * train_size
        else:
            current_train_start = train_indices[-1][1] 

        #current_train_end = current_train_start + train_size
        current_train_end = current_train_start + train_size
        train_indices.append([current_train_start, current_train_end])
        #val_start = current_train_end
        val_start = current_train_start + train_size
        val_end = current_train_end + test_size
        val_indices.append([val_start, val_end])

    return train_indices, val_indices
    

def average_cv_outputs(df, cv_iteration):
    """
    Average cross-validation outputs for features and their scores.
    Args:
        df (pd.DataFrame): DataFrame containing cross-validation outputs.
        cv_iteration (int): Number of cross-validation iterations.
    Returns:
        pd.DataFrame: A DataFrame with columns 'params_', 'num_features', 'feature', 'feature_score'.
    """
    # Combine features and scores
    features = np.array([i for i in df['features']])
    scores = np.array([i for i in df['feature_scores']])

    # Get relevant columns for grouping
    grouping_columns = ['params_', 'num_features']
    
    # Create a dictionary to store the total score and count for each feature, grouped by relevant columns
    grouped_feature_totals = {}

    # Iterate over each sample and accumulate scores for each feature, grouped by relevant columns
    for i in range(len(features)):
        key = tuple(df[grouping_columns].iloc[i])
        
        if key not in grouped_feature_totals:
            grouped_feature_totals[key] = {}
        
        for j in range(len(features[i])):
            feature = features[i][j]
            score = scores[i][j]
            
            if feature in grouped_feature_totals[key]:
                grouped_feature_totals[key][feature] += score
            else:
                grouped_feature_totals[key][feature] = score

    # Calculate average scores for each feature, grouped by relevant columns
    average_scores = {key: {feature: total / cv_iteration for feature, total in totals.items()} for key, totals in grouped_feature_totals.items()}

    # Sort features based on their average scores in descending order
    sorted_features_by_group = {key: sorted(scores.keys(), key=lambda x: average_scores[key][x], reverse=True) for key, scores in grouped_feature_totals.items()}
    result_scores_by_group = {key: [average_scores[key][feature] for feature in sorted_features_by_group[key]] for key in average_scores.keys()}

    # Create a new DataFrame with the sorted features and their average scores, grouped by relevant columns
    result_df = pd.DataFrame({
        **{col: [key[i] for key in average_scores.keys()] for i, col in enumerate(grouping_columns)},
        'feature': list(sorted_features_by_group.values()),
        'feature_score': list(result_scores_by_group.values())
    })
    
    for i in range(np.shape(result_df)[0]):
        num_features = result_df['num_features'].iloc[i]
        result_df.at[i, 'feature'] = result_df['feature'].iloc[i][:num_features].copy()
        result_df.at[i, 'feature_score'] = result_df['feature_score'].iloc[i][:num_features].copy()

    return result_df
    
def metrics_to_df(track_metrics):
    """
    Convert a dictionary containing metrics information into a DataFrame.
    Args:
        track_metrics (dict): A dictionary containing metrics information.
    Returns:
        pd.DataFrame: A DataFrame with columns 'param', 'num_features', 'score', 'y_pred'.
    """
    data = []
    for key, values in track_metrics.items():
        param, cv = key
        num_features_list = values['num_features']
        score_list = values['score']
        y_pred_list = values['y_pred']
        features_list = values['features']
        feature_scores_list = values['feature_scores']

        # Iterate through num_features, score, and y_pred lists
        for num_features, score, y_pred, features, feature_scores in zip(num_features_list, score_list, y_pred_list, features_list, feature_scores_list):
            data.append({
                'cv': cv,
                'param': param,
                'num_features': num_features,
                'score': score,
                'y_pred': y_pred,
                'features': features,
                'feature_scores': feature_scores
            })
    df = pd.DataFrame(data)

    return df

def check_column_types(df, label_cols, do_not_encode_cols):
    """ A method that checks whether the features in the dataframe given are numerical or categorical
    Args:
        df (dataframe): a dataframe containing train and validation data
        label_cols (list): a list of columns to label encode
        do_not_encode_cols (list): a list of columns to not encode
    Returns:   
        categorical_columns (list): a list of categorical features
        numerical_columns (list): a list of numerical features
    """
    categorical_columns = []
    numerical_columns = []
    # Check and return the categorical and numerical columns
    for column in df.columns:
        if (df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column])) and (column not in label_cols) and (column not in do_not_encode_cols):
            categorical_columns.append(column)
        elif (pd.api.types.is_numeric_dtype(df[column])) and (column not in label_cols) and (column not in do_not_encode_cols):
            numerical_columns.append(column)
    return categorical_columns, numerical_columns

def create_df(all_features, all_scores, all_features_rev, all_scores_rev, dataset_size, total_time, method_name, dataset_name, pred_type, feature_score, cm_val, cm_val_reversed, rebalance, rebalance_type, data_shape, is_best_score, cv_iteration):
    """ A method that creates a dataframe containing information from each model run
    Args:
        all_features (list): a list of top features subsets
        all_scores (list): a list of prediction accuracies for each feature subset
        all_features_rev (list): a list of top features subsets in reversed order
        all_scores_rev (list): a list of prediction accuracies for each feature subset with top features subsets in reversed order
        dataset_size (tuple): a tuple indicating the dimension of the original dataset
        total_time (float): the time it took the feature selection method to run
        method_name (str): the name of the feature selection method
        dataset_name (str): the name of the dataset
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        feature_score (list): the score of the features
        cm_val (dict): a dictionary containing scoring metrics
        cm_val_reversed (dict): a dictionary containing scoring metrics for top features subsets in reversed order
        rebalance (bool): a boolean indicating whether to rebalance the dataset
        rebalance_type (str): a string indicating what type of rebalancing to perform
        data_shape (dict): a dictionary containing the shapes of the train and validation datasets
        is_best_score (bool): a boolean indicating whether the run provides the optimal score
        cv_iteration (int): an integer indicating the cross validation iteration
    Returns:   
        results_df (dataframe): a dataframe containing the above information
    """
    df_len = len(all_scores)
    # Generate a dataframe containing the metrics for all feature subsets
    results_df = pd.DataFrame({
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]*df_len,
        "feature": all_features if len(all_features)> 0 else ["n/a"]*df_len,
        "score": all_scores if len(all_scores)> 0 else ["n/a"]*df_len,
        "feature_reversed" : all_features_rev if len(all_features_rev)> 0 else ["n/a"]*df_len,
        "score_reversed_rank": all_scores_rev if len(all_scores_rev)> 0 else ["n/a"]*df_len,
        "method": [method_name]*df_len,
        "dataset": [dataset_name]*df_len,
        "dataset_size": [dataset_size]*df_len,
        "runtime_sec": [total_time]*df_len,
        "prediction_type": [pred_type]*df_len,
        "feature_score": feature_score, 
        "cm_val": cm_val if len(cm_val)> 0 else ["n/a"]*df_len,
        "cm_val_reversed": cm_val_reversed if len(cm_val_reversed)> 0 else ["n/a"]*df_len,
        "rebalance": [str(rebalance)]*df_len,
        "rebalance_type": [rebalance_type]*df_len,
        "data_shape": [data_shape]*df_len,
        "is_best_score": [str(x) for x in is_best_score],
        "cv_iteration": [cv_iteration]*df_len
    })
    return results_df

def add_to_dataframe(df, import_name, export, export_name=None):
    """ A method for appending model results to any exisiting tracked outputs
    Args:
        df (dataframe): a dataframe containg model results
        import_name (str): name of the file containing the model results
        export (bool): a boolean to indicate whether to export the dataframe into an Excel file
        export_name (bool): name of the export file
    Returns:
        merged_results_import_updated (dataframe): a dataframe containing newly added model results
    
    """
    # Read the import file
    merged_results_import = pd.read_excel(f"./{import_name}")
    merged_results_import.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    # Add index to the dataframe
    index_df = pd.DataFrame({"index" : np.arange(0, np.shape(df)[0]).tolist()})
    df_indexed_df = pd.concat([index_df, df], axis=1)

    #print("Shape (Imported df): ", np.shape(merged_results_import))
    #display(merged_results_import.head())

    #print("Shape (df): ", np.shape(df_indexed_df))
    #display(df_indexed_df.head())

    # Merge dataframes, convert feature names to string, remove duplicates
    merged_results_import_updated = pd.concat([merged_results_import, df_indexed_df])
    merged_results_import_updated['feature'] = merged_results_import_updated['feature'].map(str)
    merged_results_import_updated['feature_reversed'] = merged_results_import_updated['feature_reversed'].map(str)
    merged_results_import_updated = merged_results_import_updated.drop_duplicates()

    if export:
        merged_results_import_updated.to_excel(f"./{export_name}.xlsx", index=False)
        print(f"Exported: {export_name}.xlsx")
    return merged_results_import_updated

def rebalance_data(data, rebalance_type, seed):
    """ A method for rebalancing a dataset with imbalanced target values
    Args:
        data (dict): a dictionary containing train and validation data
        rebalance_type (str): a string indicating which rebalancing method to use
        seed (int): a random state
    Returns:   
        data (dict): a dictionary containing rebalanced train data
    """
    print(f"X_train shape before resmapling. X_train: {np.shape(data['X_train'])} y_train: {np.shape(data['X_train'])}")
    print(f"y_train classes before resampling: {dict(Counter(data['y_train']))}")

    X_data_df = data['X_train']
    y_data_df = data['y_train']

    # Various methods for resampling
    if rebalance_type.lower() == "random_over_sampler":
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=seed)
        oversampler.fit(X_data_df, y_data_df)
        X_resampled, y_resampled = oversampler.fit_resample(X_data_df, y_data_df)
        
    if rebalance_type.lower() == "smoten":
        sampler = SMOTEN(random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    if rebalance_type.lower() == "smote":
        sampler = SMOTE(random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    if rebalance_type.lower() == "smotenc":
        sampler = SMOTENC(random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    if rebalance_type.lower() == "borderlinesmote":
        sampler = BorderlineSMOTE(random_state=seed)        
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)
        
    if rebalance_type.lower() == "adasyn":
        sampler = ADASYN(random_state=seed)         
        X_resampled, y_resampled = sampler.fit_resample(X_data_df, y_data_df)

    data['X_train'] = X_resampled
    data['y_train'] = y_resampled

    print(f"X_train shape after resmapling. X_train: {np.shape(data['X_train'])} y_train: {np.shape(data['X_train'])}")
    print(f"y_train classes after resampling: {dict(Counter(data['y_train']))}")
    return data

def map_data(input_data, map_cols, prev_col_name, mapped_col_names):
    """Map the input data based on a mapping dictionary
    Args:
        input_data (dataframe): The input dataset
        map_cols (list): A list of columns to be mapped
        prev_col_name (list): A list of the names of the columns to be mapped
        mapped_col_names (list): A list of the mapped columns' new names
    Returns:
        A new dataset with newly mapped columns
    """
    # Generate a dictionary containing the above mappings, the previous and new column names
    mapping_dict = {}
    for m, m_col_name, p_col_name in zip(map_cols, mapped_col_names, prev_col_name):
        mapping_dict[m_col_name] = (p_col_name, m)
    # Map new columns
    df_copy = copy.deepcopy(input_data)
    df_target = df_copy.iloc[:, -1]
    df_copy = df_copy.iloc[:, :-1]
    for k,v in mapping_dict.items():
        df_copy[k] = df_copy[v[0]].map(v[1])
    # Drop previous columns    
    df_copy = df_copy.drop(columns=[v[0] for v in mapping_dict.values()])       
    return pd.concat([df_copy, df_target], axis=1)

def tune_cv_split(data, min_test_val_size=50, val_test_prop_constraint=0.2, num_split_constraint=3):
    """Create various combinations of cv splits for cross-validation. The default constraints are defined such that the 
       number of instances in the validation set is about 20% the size of the training set.
    Args:
        data (data_frame): A dataframe containing the train, validation, and test data
        min_test_val_size (int): An integer specifying the minimum number of instances in the validation and test set
        val_test_prop_constraint (float): A float indicating the proportional size of the val/test set relative to the train set
        num_split_constraint (int): An integer specifying the minimum number of train/val splits required
    Returns:
        A list containing lists of train and test size, and number of splits
    """
    # Number of instances in the dataset
    sample_size = np.shape(data)[0]
    # Generate the possible val/test set size
    n_iter = int((sample_size - 2*min_test_val_size)*val_test_prop_constraint / min_test_val_size)
    test_val_size_list = []
    for n in range(1, n_iter):
        test_val_size = min_test_val_size*n
        test_val_size_list.append(test_val_size)
    # Generate the possible final combinations for train/val/test size
    train_test_list = []
    for train_val_size in test_val_size_list:
        # Determine the number of splits possible
        num_split = int(np.floor((sample_size - 2*train_val_size) / (train_val_size / val_test_prop_constraint)))
        # If the calculated number of split doesn't meet num_split_constraint (the min specified)
        if num_split < num_split_constraint: 
            break  
        train_size = int(train_val_size / val_test_prop_constraint)
        train_test_list.append([train_size, train_val_size, num_split])
    print(train_test_list)
    return train_test_list

def convert_to_sample(path_in, path_out, filename_in, filename_out, date_threshold, filter_type, stocks=[]):
    """
    Filter a CSV file based on a date threshold and save the filtered data to a new CSV file.

    Args:
        path_in (str): The path to the input CSV file.
        path_out (str): The path to the output CSV file where the filtered data will be saved.
        date_threshold (str): The date threshold for filtering the data. Rows with a date greater than or equal to this threshold will be included in the output.
        filter_type (str): A string indicating how the data should be filtered
    """
    if len(stocks) > 0 and filter_type == "individual stocks":
        for stock in stocks:
            # Create a CSV writer for the output file
            with open(path_out + stock + "_" + filename_out, 'w', newline='') as output_file:
                csv_writer = csv.writer(output_file)

                # Write the header to the output file
                header_written = False

                # Use pandas to read the CSV file in chunks
                chunksize = 1000  # Adjust the chunk size based on your available memory
                for chunk in pd.read_csv(path_in + filename_in, chunksize=chunksize):

                    # Filter rows where the date is greater than the threshold
                    chunk = chunk[(chunk['date'] >= date_threshold) & (chunk['ticker'] == stock)]

                    # Write the chunk to the output file
                    chunk.to_csv(output_file, header=not header_written, index=False, mode='a')

                    # Ensure the header is only written once
                    header_written = True
            print(f"Filtered data has been saved to '{path_out + stock + '_' + filename_out}'.")

    else:
        # Create a CSV writer for the output file
        with open(path_out + filename_out, 'w', newline='') as output_file:
            csv_writer = csv.writer(output_file)

            # Write the header to the output file
            header_written = False

            # Use pandas to read the CSV file in chunks
            chunksize = 1000  # Adjust the chunk size based on your available memory
            for chunk in pd.read_csv(path_in + filename_in, chunksize=chunksize):

                # Filter rows where the date is greater than the threshold
                if filter_type == "combined stocks":
                    chunk = chunk[(chunk['date'] >= date_threshold) & (chunk['ticker'].isin(stocks))]
                else:
                    chunk = chunk[chunk['date'] >= date_threshold]

                # Write the chunk to the output file
                chunk.to_csv(output_file, header=not header_written, index=False, mode='a')

                # Ensure the header is only written once
                header_written = True
        print(f"Filtered data has been saved to '{path_out + filename_out}'.")

def create_time_feature(df):
    """
    Create time-related features from the 'date' column of a DataFrame.

    Args:
    - df (pd.DataFrame): Input DataFrame containing a 'date' column.

    Returns:
    pd.DataFrame: DataFrame with additional time-related features.
    """
    # Convert the 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Extract various time-related features
    df['dayofmonth'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.weekofyear

    return df

def setup_seed(seed):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

