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
train_data, test_data, train_labels, test_labels = preprocess_and_spilt(data, window_size, overlap_percentage,0.1)

X_train_flat = np.array([window.reshape(-1) for window in train_data])
X_test_flat = np.array([window.reshape(-1) for window in test_data])

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, train_labels)
rf_predicitions = rf_model.predict(X_test_flat)
rf_accuracy = accuracy_score(test_labels, rf_predicitions)

print(f"Random Forest Classifier Accuracy: {rf_accuracy * 100:.2f}")

# train_data = torch.from_numpy(train_data).float()
# test_data = torch.from_numpy(test_data).float()

# train_labels = torch.from_numpy(train_labels).long()
# test_labels = torch.from_numpy(test_labels).long()

# # convert them into PyTorch Datasets 
# train_data = TensorDataset(train_data, train_labels)
# test_data = TensorDataset(test_data, test_labels)

# # Translate into dataloader objects
# batchsize = 16
# train_loader = DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)
# test_loader = DataLoader(test_data, batch_size=batchsize)

# class HARModel(nn.Module):
#     # New class inherited from the nn.module
#     def __init__(self, input_features, num_classes):
#         super(HARModel, self).__init__() # `super()` allows python to access function from nn.module (parent class)
        
#         self.conv1=nn.Conv1d(
#             in_channels=input_features,
#             out_channels=64,
#             kernel_size=5,
#             padding=2
#         )
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(64 * 250, 128)
#         self.fc2 = nn.Linear(128, num_classes) # num_classes = 2 (sitting, walking)

#         # input layer
#         # self.input = nn.Linear(6, 16)

#         # hidden layer 8 minutes and 23 seconds
#         # self.hidden1=nn.Linear(16, 32)
#         # self.hidden2=nn.Linear(32, 32)

#         # # output layer
#         # self.output = nn.Linear(32, 1)

#     def forward(self, x):
#         x=x.permute(0,2,1)

#         # Forward pass through the convolutional layers
#         x = self.conv1(x)
#         x = self.relu(x)
        
#         # Flatten the output for the fully connected layers
#         x = self.flatten(x)
        
#         # Forward pass through the fully connected layers
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
        
#         return x
#         # x = F.relu(self.input(x))
#         # x = F.relu(self.hidden1(x))
#         # x = F.relu(self.hidden2(x))
#         # return self.output(x)

# numepochs=500

# def trainTheModel():
#     lossfun=nn.CrossEntropyLoss()
#     optimizer=torch.optim.Adam(HARnet.parameters(),lr=0.0005)

#     # initialize loss
#     losses=torch.zeros(numepochs)
#     trainAcc=[]
#     testAcc=[]

#     # loop over epochs
#     for epochi in range(numepochs):
#         HARnet.train()  # sets the model to training mode

#         # loop over training data batches
#         batchAcc=[]
#         batchLoss=[]

#         for x, y in train_loader:
#             # forward pass and losses
#             yHat = HARnet(x)
#             loss = lossfun(yHat,y)

#             # back prop
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # losses from this batch
#             batchLoss.append(loss.item())

#             # compute training accuracy for this batch
#             _, predicted = torch.max(yHat.data, 1)
#             batchAcc.append( 100 * (predicted == y).float().mean().item() )
        
#         # now that we've trained through the batches, get their average training accuracy
#         trainAcc.append( np.mean(batchAcc) )
#         # end of batch loop
#         losses[epochi] = np.mean(batchLoss)

#         ''' # test accuracy
#         HARnet.eval()
#         x,y = next(iter(test_loader)) # extract X, y from test dataloader
#         with torch.no_grad(): # deactivates auto grad
#             yHat=HARnet(x)
#         testAcc.append( 100*torch.mean(((yHat>0) == y).float()).item())'''

#          # Correct way to calculate test accuracy
#         HARnet.eval()
#         correct_predictions = 0
#         total_samples = 0
#         with torch.no_grad():
#             for x_test, y_test in test_loader:
#                 outputs_test = HARnet(x_test)
#                 _, predicted_test = torch.max(outputs_test.data, 1)
#                 total_samples += y_test.size(0)
#                 correct_predictions += (predicted_test == y_test).sum().item()
        
#         testAcc.append(100 * correct_predictions / total_samples)
    
#     # function output
#     return trainAcc, testAcc, losses

# startTime=time.time()
# input_features = 6
# num_classes = 2
# HARnet=HARModel(input_features, num_classes)
# trainAcc,testAcc,losses = trainTheModel()

# endTime=time.time()
# print(f"your model train for {int((endTime-startTime) // 60)} minutes and {int((endTime-startTime) % 60)} seconds")

# fig, ax = plt.subplots(1,2,figsize=(6,3))

# ax[0].plot(trainAcc)
# ax[0].set_title('training accuracy (%)')
# ax[0].set_xlabel('percentage')
# ax[0].set_xlabel('epochs')

# ax[1].plot(testAcc)
# ax[1].set_title('test accuracy (%)')
# ax[1].set_xlabel('percentage')
# ax[1].set_xlabel('epochs')

# plt.show()

