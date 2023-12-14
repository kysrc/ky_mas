# Author: JYang
# Last Modified: Dec-07-2023
# Description: This script provides the method(s) for generating visualizations

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter, OrderedDict
import pandas as pd
from IPython.display import display
import numpy as np

class plotScore:
    """ A class for generating visualizations
    Args:
        data (list): a list of data points
        feature_impt (list): a list of feature names
        pred_type (str): a string indicating the type of prediction problem: classification or regression
        use_num_features(list): a list containing the number of features selected
    """
    def __init__(self, data, feature_impt, pred_type, use_num_features):
        self.data = data
        self.feature_impt = feature_impt
        self.pred_type = pred_type.lower()
        self.use_num_features = use_num_features

    def score_plot(self):
        """ A method that displays a bar plot of scores with labels """
        n = len(self.data)
        bar_positions = list(range(1, n + 1))
        plt.bar(bar_positions, self.data, align='center', edgecolor='black')
        plt.xlabel("Number of Features")
        score_type = "F1" if self.pred_type == "classification" else "RMSE"
        plt.ylabel(f"{score_type} Score")
        plt.title(f"{score_type} Score by Number of Features")
        
        # Set x-axis labels based on feature names
        x_labels = self.use_num_features
        plt.xticks(bar_positions, labels=x_labels)
        plt.show()
        
        for f in self.use_num_features: print(f"\nThe {f} features are: {self.feature_impt[:f]}\n")
            
class plotCurve:
    """ A class for stacking multiple plots together
    Args:
        data (list): a list of data points
        label (list): a list of labels for the legend
        ds_title (str): a title for the plot
        pred_type (str): a string indicating the type of prediction problem: classification or regression
    """
    def __init__(self, data, label, ds_title, pred_type):
        self.data = [data]
        self.label = [label]
        self.ds_title = ds_title
        self.pred_type = pred_type.lower()

    def plot_line(self):
        """ Generate a plot"""
        for i, data in enumerate(self.data):
            plt.plot(range(1, len(data) + 1), data, label=f"Line {i+1}")

        plt.xlabel("Number of Features")
        score_type = "F1" if self.pred_type == "classification" else "RMSE"
        plt.ylabel(f"{score_type} Score")
        plt.title(f"{score_type} Score over Number of Features [{self.ds_title}]")
        n = 5
        x_ticks = range(1, len(self.data[0]) + 1, n)
        plt.xticks(x_ticks)
        plt.legend(self.label, bbox_to_anchor=(1.5, 1), loc='upper right')
        plt.show()

    def add_line(self, data, label):
        """ Add a line to an existing plot
        Args:
            data (list): a list of data points
            label (list): a list of labels for the legend
        """
        if isinstance(data, (int, float)):
            data = [data]
        self.label.append(label)
        self.data.append(data)
        self.plot_line()

def word_cloud_freq(file_path, top_n=10):
    """Generate a wordcloud and histogram using feature frequencies
    Args:
        file_path (str): a string indicating the path where the outputs file is stored
        top_n (int): an integer that determines the top number of features to display in the histogram
    """
    # Import data
    outputs_df = pd.read_excel(file_path, sheet_name="data")
    # Filter for optimal feature subsets only
    feature_df = pd.DataFrame(outputs_df.loc[outputs_df['is_max_acc'] == True]['feature']).reset_index(drop=True)
    # Append features to a list
    feature_list = []
    for i in range(len(feature_df)):
        for s in ast.literal_eval(feature_df['feature'][i]):
            feature_list.append(s)
    # Get feature counts
    feature_cnt = Counter(feature_list)
    # Order feature counts
    sorted_word_counts = OrderedDict(feature_cnt.most_common())
    
    # Generate wordcloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(sorted_word_counts)
    # Plot wordcloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    # Sort word counts
    sorted_word_counts_list = list(sorted_word_counts)
    # Filter for top n features
    words, frequencies = zip(*list(sorted_word_counts.items())[:top_n])
    
    # Create a histogram
    plt.bar(words, frequencies)
    plt.title(f'Top {top_n} Word Frequency Histogram')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate x-labels for better readability
    plt.tight_layout()
    plt.show()

def plot_ts(df):
    """Plot the predictions and true values of a time series
    Args:
    df (dataframe): a datafrome containing predictions and actual values of a time series
    """
    plt.figure(figsize=(4, 4))
    plt.scatter(np.arange(0, len(df['y_true'])), df['y_true'], label='Average y_true', marker='o', color='blue')
    plt.scatter(np.arange(0, len(df['y_true'])), df['y_pred'], label='Average y_pred', marker='x', color='red')
    plt.xlabel('Time (Date Converted to Integer Value)')
    plt.ylabel('Average Returns')
    plt.title('Scatter Plot of Average y_true and y_pred over Time')
    plt.legend()
    plt.show()


