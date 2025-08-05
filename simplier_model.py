# for DP modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# for number - crunching
import numpy as np
import scipy.stats as stats

# for dataset management
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns

# for file mangement
import glob
import os
import time
from typing import List, Dict, Any

def sliding_window(data: pd.DataFrame, window_size: int, overlap_percentage: float):
    # Data columns to be segmented
    sensor_cols=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

    # Step size based on window size and overlap
    step_size = int(window_size * (1-overlap_percentage))

    if step_size <= 0:
        raise ValueError("calculated step size is zero or negative")
    
    windows_with_labels=[]

    # Iterate through the df in steps
    for i in range(0, len(data) - window_size + 1, step_size):
        window       = data.iloc[i:i+window_size]
        # adding [0]???
        window_data = window[sensor_cols].values
        window_label = window['activity_label'].mode()[0]

        windows_with_labels.append({'data':window_data, 'activity_label': window_label})

    # print(len(window_label))
    return windows_with_labels

def preprocess_and_spilt(data: pd.DataFrame, window_size: int, overlap_percentage: float, test_size:float=0.2):
    raw_segments=sliding_window(data,window_size,overlap_percentage)

    # Separate raw data and labels for further processing
    X_raw = [s['data'] for s in raw_segments]
    y_labels = [s['activity_label'] for s in raw_segments]
    # print(f"the type is {type(y_labels[0])}")

    X_standardized=[]
    # Initializes the Standard Scaler, it will be fit for each window.
    # The scaler should only see the data it is transforming to avoi data leaks from the future
    scaler=StandardScaler()

    for segment in X_raw:
        # segment is a 2D array
        standardized_segment = scaler.fit_transform(segment)
        X_standardized.append(standardized_segment)
    # converts the list to an np array
    X_standardized = np.array(X_standardized)

    label_to_int = {'sitting': 0, 'walking': 1}
    y_encoded = np.array([label_to_int[i] for i in y_labels])

    train_data, test_data, train_labels, test_labels = train_test_split(X_standardized, y_encoded, test_size=test_size,stratify=y_encoded, random_state=42)
    return train_data, test_data, train_labels, test_labels

# read data from the combine csv
data = pd.read_csv('combine_mpu9250.csv', sep=',')

window_size=250
overlap_percentage=0.3
train_data, test_data, train_labels, test_labels = preprocess_and_spilt(data, window_size, overlap_percentage,0.2)

X_train_flat = np.array([window.reshape(-1) for window in train_data])
X_test_flat = np.array([window.reshape(-1) for window in test_data])

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import List, Dict, Any

# lr_model = LinearRegression()
# lr_model.fit(X_train_flat, train_labels)
# lr_predictions = lr_model.predict(X_test_flat)

# # To calculate 'accuracy', we need to convert predictions to class labels.
# # We'll use a threshold of 0.5.
# lr_predictions_classes = (lr_predictions > 0.5).astype(int)
# lr_accuracy = accuracy_score(test_labels, lr_predictions_classes)
# lr_mse = mean_squared_error(test_labels, lr_predictions)

# print(f"Linear Regression 'Accuracy' (with 0.5 threshold): {lr_accuracy*100:.4f}")
# print(f"Linear Regression Mean Squared Error: {lr_mse*100:.4f}")

iterations=10

rf_accuracy_list = []
for i in range(iterations):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_model.fit(X_train_flat, train_labels)
    rf_predicitions = rf_model.predict(X_test_flat)
    rf_accuracy = accuracy_score(test_labels, rf_predicitions)
    print(f"Random Forest Classifier Accuracy: {rf_accuracy * 100:.4f}")
    rf_accuracy_list.append(rf_accuracy_list)

# [print(f"Random Forest Classifier Accuracy: {rf_acc}") for rf_acc in rf_accuracy_list]