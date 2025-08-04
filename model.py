# for DP modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

# read data from the combine csv
data = pd.read_csv('combine_mpu9250.csv', sep=',')
print(data.describe())
print(data.head())
# # sliding window to create segment of value data

# # z-score all variables except for activity_label
# cols2zscore=data.keys()
# cols2zscore = cols2zscore.drop('activity_label')

# # normalizes data
# data[cols2zscore] = data[cols2zscore].apply(stats.zscore)

# data['bool_activity']=0
# data['bool_activity'][data['activity_label']=='walking'] = 1

# # print(data[['bool_activity','activity_label']])

# # # convert from pandas dataframe to tensor
# dataT = torch.tensor(data[cols2zscore].values).float()
# labelT = torch.tensor(data['bool_activity'].values).float()
# print(f"the data tensor shape is {dataT.shape}\nthe label tensor shape is {labelT.shape}")

# # labels need to be a "tensor"
# labelT = labelT[:, None]
# print(f'new shape is {labelT.shape}')

# # use scikitlearn to split the data
# train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelT, test_size=.1)

# # convert them into PyTorch Datasets 
# train_data = TensorDataset(train_data, train_labels)
# test_data = TensorDataset(test_data, test_labels)

# # Translate into dataloader objects
# batchsize = 32
# train_loader = DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)
# test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

# class HARModel(nn.Module):
#     # New class inherited from the nn.module
#     def __init__(self):
#         super().__init__() # `super()` allows python to access function from nn.module (parent class)
#         # input layer
#         self.input = nn.Linear(6, 16)

#         # hidden layer 8 minutes and 23 seconds
#         self.hidden1=nn.Linear(16, 32)
#         self.hidden2=nn.Linear(32, 32)

#         # output layer
#         self.output = nn.Linear(32, 1)

#     def forward(self, x):
#         x = F.relu(self.input(x))
#         x = F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         return self.output(x)

# numepochs=500

# def trainTheModel():
#     lossfun=nn.BCEWithLogitsLoss()
#     optimizer=torch.optim.SGD(HARnet.parameters(),lr=0.1)

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
#             batchAcc.append( 100*torch.mean(((yHat>0) == y).float()).item() )
        
#         # now that we've trained through the batches, get their average training accuracy
#         trainAcc.append( np.mean(batchAcc) )
#         # end of batch loop
#         losses[epochi] = np.mean(batchLoss)

#         # test accuracy
#         HARnet.eval()

#         x,y = next(iter(test_loader)) # extract X, y from test dataloader
#         with torch.no_grad(): # deactivates auto grad
#             yHat=HARnet(x)

#         testAcc.append( 100*torch.mean(((yHat>0) == y).float()).item())

#     # function output
#     return trainAcc, testAcc, losses

# startTime=time.time()

# HARnet=HARModel()
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

        