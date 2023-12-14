<p align="center">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/flow_diagram.png" width="1200" />
</p>

---

This project involves the development of a Python pipeline encompassing data ingestion, feature selection, model optimization, and prediction using the XGBoost algorithm. It emphasizes adaptability in selecting variable numbers of features, considering computational costs, eliminating irrelevant features, and mitigating overfitting. The pipeline's foundation lies in a thorough literature review that identifies and implements literature state-of-the-art feature selection methods, as well as traditional feature selection methods that are compatible with the XGBoost architecture. The dataset encompasses both standard datasets and a custom financial time series dataset created to simulate real-world conditions. The pipeline integrates rebalancing techniques and data transformation processes. The goal is to achieve robust model performance through back testing and hyperparameter tuning, paving the way for continuous improvement and future iterations of the project.

---

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/py_ver.png" width="150" />
</p>


## Pseudocode

A high-level view for the pipeline processes.

<p align="center">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/pseudocode.png" width="1200" />
</p>

---

## Rolling Window Back Testing

A view of a single back testing window broken out by its train-validation splits. 

<p align="center">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/rolling_bt.png" width="1200" />
</p>

---

## Feature Score Aggregation

How feature scores are generalized (feature scores averaged) across multiple train-validation splits.

<p align="center">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/feature_selection_example.png" width="1200" />
</p>

---

## Train-Validation Splits (All Features)

A view of a single back testing window with the train-validation splits and scoring metrics for different combinations of hyperparameters. The same concept applies for the top 50 subset features.

<p align="center">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/bt_all_features.png" width="1200" />
</p>

---

## Notebook

Import the necessary libraries in the requirements.txt file.

Change the file directory to the path where the feature_selection_timeseries folder is located.

```python
import os
directory_path = input("Enter your file directory: ")
os.chdir(directory_path)
```

Import other necessary libraries.

```python
from feature_selection_timeseries.src.models.pipeline import run
from feature_selection_timeseries.src.models.utils import create_time_feature, tune_cv_split, convert_to_sample 
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import csv
```

Defining the file names and paths where the original data is stored and where the data filtered by the date threshold will be saved to.

Alternatively, using the 11 stocks files, which can be downloaded via the URL link below (request for access is required):

https://drive.google.com/drive/folders/1yN-JTu9pvL8Tm2xTSiK5L8F_L1tIWxoQ?usp=sharing

After completing the downloads, add the files to the directory: ./feature_selection_timeseries/data/raw/sp500_subset/

```python
year = "2006"
date_threshold = year + '-01-01'  # filter for data with date >=
sub_folder = "sp500_subset"

# path of the file containing the features
x_filename_in = 'stock_x.csv'
x_filename_out = f'stock_x_sample_regression_{year}_filtered_11_stocks.csv'
x_in_path = f'./feature_selection_timeseries/data/raw/{sub_folder}/'
x_out_path = f'./feature_selection_timeseries/data/raw/{sub_folder}/'

# path of the file containing the label
y_filename_in = 'stock_y_ret84.csv'
y_filename_out = f'stock_y_ret84_sample_regression_{year}_filtered_11_stocks.csv'
y_in_path = x_in_path
y_out_path = x_out_path
```

If you need to filter the original data by certain stocks, add the stock tickers to the list below and run the convert_to_sample() function.

```python
stocks=[
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"
]

convert_to_sample(
    path_in=x_in_path,
    path_out=x_out_path,
    filename_in = x_filename_in,
    filename_out = x_filename_out,
    date_threshold=date_threshold,
    filter_type="combined stocks",
    stocks=stocks
)

convert_to_sample(
    path_in=y_in_path,
    path_out=y_out_path,
    filename_in = y_filename_in,
    filename_out = y_filename_out,
    date_threshold=date_threshold,
    filter_type="combined stocks",
    stocks=stocks
)
```

Import the files containing the features and target values. Then create a set of new date features. Finally merge the features and label into a single dataframe.

```python
# Specify the path to your CSV file
x_path = f'{x_in_path}{x_filename_out}'
y_path = f'{y_in_path}{y_filename_out}'
# Import data
sp500_df = pd.read_csv(x_path)
y = pd.read_csv(y_path)
# Adjust labels
y['target'] = y['ret_fwd_84']
# Create additional time features
sp500_df = create_time_feature(sp500_df)
# Combine features and target
sp500_df = pd.concat([sp500_df.iloc[:, 1:], y['target']], axis=1)
```

As an illustration, to keep execution time manageable a single stock will be used in this example. Additionally, two sets of hyperparameters and five feature selection methods (XGBoost, Permutation, SHAP, Cart, and Lasso) are considered. Run the codes below to filter for the ticker ADBE (Adobe Inc).  

```python
sp500_df = sp500_df[sp500_df['ticker'] == 'ADBE']
y = y[y['ticker'] == 'ADBE']
```

In the train_model.py file.

```python
param_grid = {
   'min_child_weight': [0.5, 1],
   'gamma': [0.01],  
   'max_depth': [6], 
   'learning_rate': [0.3], 
   'alpha': [0]  
}   
```

View the updated dataframes.

```python
display(y.head())
print(f"Dataset Shape: {np.shape(y)}")
```
<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/adbe_table_y.png" width="300" />
</p>


```python
display(sp500_df.head())
print(f"Dataset Shape: {np.shape(sp500_df)}")
```
<p align="center">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/adbe_table_x.png" width="1200" />
</p>

Initializing the pipeline.

Compute the list of possible splits and their corresponding train-validation sizes. 5-fold splits will be used in this example, as specified. Note that the first term is not used, as the rolling window size and train-validation split size are computed separately, but the validation/test size is indicated by the middle term, and the number of splits for both the back test rolling window and train-validation parts is the same.

```python
# Possible train validation splits
train_test_list = [tune_cv_split(
    sp500_df.iloc[-np.shape(sp500_df)[0]:,:],
    val_test_prop_constraint = 0.2, # Size of validation set relative to the train set
    num_split_constraint = 5 # Number of splits
)[-1]]

keep_data_index = train_test_list[0][0]*train_test_list[0][2] + 2*train_test_list[0][1]
print(f"\nUsing Split: {train_test_list}")
```

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/train_val_split.png" width="300" />
</p>


Initialize the other arguments.

```python
r1 = run(
    cross_validation_type= "moving window", # or "expanding window"
    save_output_file = True, # Whether to save test outputs
    raw_df = sp500_df.iloc[-keep_data_index:, :].reset_index(drop=True), # Discard extra data instances from the beginning of the time series rather than the end
    y = y.iloc[-keep_data_index:, :].reset_index(drop=True), # Discard extra data instances from the beginning of the time series rather than the end
    train_test_list = train_test_list, # A list of possible list of train and validation size, and number of splits
    methods = ["xgboost", "permutation", "shap", "lasso", "cart"], # Available methods: ["xgboost", "cae", "permutation", "shap", "boruta", "sage", "lasso", "cart", "svm", "rf", "stg", "dynamic"]
    rebalance_type = ["None"], # ["borderlinesmote", "smoten", "random_over_sampler", "smote", "None"]
    label_cols = [], # Columns to label encode
    do_not_encode_cols = ["dayofmonth", "dayofweek", "quarter", "month", "year", "dayofyear", "weekofyear"], # These fields are not transformed
    seed = 42,
    target_colname = "target", # The name of the field that holds the true values
    dataset_name = "sp500",
    pred_type = "regression",
    append_to_full_df = False,
    n_features = 50,  # The number of top features to filter for
    feature_direction = "top", # Feature order based on their scores in descending order
    train_outputs_file_name = None,
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    scaler_filename = "./feature_selection_timeseries/data/processed/scaler_saved.save",
    encoder_filename = "./feature_selection_timeseries/data/processed/encoder_saved.save",
    label_encoder_filename = "./feature_selection_timeseries/data/processed/lalbel_encoder_saved.save",
    test_output_file_name = f"./feature_selection_timeseries/data/experiment/Consolidated_Stocks_FS_timeseries_sp500_outputs_test_results_",
    test_pred_file_name = f"./feature_selection_timeseries/data/experiment/Consolidated_Stocks_FS_timeseries_sp500_outputs_test_preds_",
    print_outputs_train = True,
    print_outputs_test = True
)
```

Train the model. Average the feature scores and validation scores, and determine the optimal hyperparameters. 

The back testing rolling windows are: 

```python
"""
Train: [[0, 3300], [150, 3450], [300, 3600], [450, 3750], [600, 3900]]
Test:  [[3300, 3450], [3450, 3600], [3600, 3750], [3750, 3900], [3900, 4050]]
"""
```

Within each rolling window, there are 5-fold train-validation splits. 

```python
"""
1. Train:      [[0, 630], [630, 1260], [1260, 1890], [1890, 2520], [2520, 3150]]
   Validation: [[630, 780], [1260, 1410], [1890, 2040], [2520, 2670], [3150, 3300]]

2. Train:      [[150, 780], [780, 1410], [1410, 2040], [2040, 2670], [2670, 3300]]
   Validation: [[780, 930], [1410, 1560], [2040, 2190], [2670, 2820], [3300, 3450]]

3. Train:      [[300, 930], [930, 1560], [1560, 2190], [2190, 2820], [2820, 3450]]
   Validation: [[930, 1080], [1560, 1710], [2190, 2340], [2820, 2970], [3450, 3600]]

4. Train:      [[450, 1080], [1080, 1710], [1710, 2340], [2340, 2970], [2970, 3600]]
   Validation: [[1080, 1230], [1710, 1860], [2340, 2490], [2970, 3120], [3600, 3750]]

5. Train:      [[600, 1230], [1230, 1860], [1860, 2490], [2490, 3120], [3120, 3750]]
   Validation: [[1230, 1380], [1860, 2010], [2490, 2640], [3120, 3270], [3750, 3900]]
"""
```

Train models and obtain the optimal hyperparamters.

```python
r1.train()
```

Set the optimal hyperparameters and apply the optimal models on the hold out test data.

```python
r1.test()
```

**Test Results:**

CART 50 Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/cart_test_50_bt.png" width="1200" />
</p>

CART All Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/cart_test_all_bt.png" width="1200" />
</p>

LASSO 50 Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/lasso_test_50_bt.png" width="1200" />
</p>

LASSO All Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/lasso_test_all_bt.png" width="1200" />
</p>

Permutation 50 Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/permu_test_50_bt.png" width="1200" />
</p>

Permutation All Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/permu_test_all_bt.png" width="1200" />
</p>

SHAP 50 Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/shap_test_50_bt_v2.png" width="1200" />
</p>

SHAP All Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/shap_test_all_bt_v2.png" width="1200" />
</p>

XGBoost 50 Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/xgb_test_50_bt.png" width="1200" />
</p>

XGBoost All Features

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/xgb_test_all_bt.png" width="1200" />
</p>

Average Test Scores (RMSE) of All Back Testing Windows For Each Method and Number of Features:

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/avg_test_rmse_all_bt_windows_v2.png" width="600" />
</p>

From the average test scores (RMSE) of all back testing windows for each method, it's evident that the SHAP feature selection method generated better subset of features than the other methods in this experiment.

---

## Sources

1. *SAGE:* Ian Covert, Scott Lundberg, Su-In Lee. "Understanding Global Feature Contributions With Additive Importance Measures." NeurIPS 2020 <https://github.com/iancovert/sage>

2. *Dynamic:* Ian Covert, Wei Qiu, Mingyu Lu, Nayoon Kim, Nathan White, Su-In Lee. "Learning to Maximize Mutual Information for Dynamic Feature Selection." ICML, 2023. <https://github.com/iancovert/dynamic-selection>

3. *CAE:* Abid, Abubakar, Muhammad Fatih Balin, and James Zou. "Concrete autoencoders for differentiable feature selection and reconstruction." arXiv preprint arXiv:1901.09346 (2019). <https://github.com/mfbalin/Concrete-Autoencoders/tree/master>

4. *STG:* Yamada, Yutaro, et al. "Feature selection using stochastic gates." International Conference on Machine Learning. PMLR, 2020 <https://github.com/runopti/stg>

5. *XGBoost:* Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016. <https://xgboost.readthedocs.io/en/stable/index.html>

6. *SHAP:* Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 (2017). <https://github.com/shap/shap>

7. *Boruta:* Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11 <https://github.com/scikit-learn-contrib/boruta_py>

8. *Permutation:* <https://scikit-learn.org/stable/modules/permutation_importance.html>

9. *LASSO:* <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>

10. *CART:* <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>

11. *SVM:* <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>

12. *RF:* <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
