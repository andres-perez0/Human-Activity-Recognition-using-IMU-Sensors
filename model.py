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

def combineCSV():
    relativePath='IMU_Data'

    allFile = glob.glob(os.path.join(relativePath, '*.csv')) 
    df_list=[]

    for fN in allFile:
        # print(fN)
        df = pd.read_csv(fN, header=0,on_bad_lines='skip')
        df_list.append(df)

    combine_df = pd.concat(df_list,axis=0,ignore_index=True)
    combine_df.to_csv('combine_mpu9250.csv', index=False,encoding='utf-8)')

# print(data.head())
# print(data.describe())

# list number of unique values per columns
# for i in data.keys():
#     print(f"{i} has {len(np.unique(data[i]))} unique values")


'''
# pairwise plots
cols2plot=['activity_label','accX',  'accY',  'accZ']
#  'gyroX',  'gyroY',  'gyroZ'
sns.pairplot(data[cols2plot],kind='reg',hue='activity_label')
plt.show()
'''

'''fig,ax = plt.subplots(1,figsize=(17,4))
ax = sns.boxplot(data=data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()'''

# read data from the combine csv
data = pd.read_csv('combine_mpu9250.csv', sep=',')

# z-score all variables except for activity_label
cols2zscore=data.keys()
cols2zscore = cols2zscore.drop('Timestamp')
cols2zscore = cols2zscore.drop('activity_label')

# normalizes data
data[cols2zscore] = data[cols2zscore].apply(stats.zscore)

data['bool_activity']=0
data['bool_activity'][data['activity_label']=='walking'] = 1

# print(data[['bool_activity','activity_label']])

# # convert from pandas dataframe to tensor
dataT = torch.tensor(data[cols2zscore].values).float()
labelT = torch.tensor(data['bool_activity'].values).float()
# print(f"the data tensor shape is {dataT.shape}\nthe label tensor shape is {labelT.shape}")

# labels need to be a "tensor"
labelT = labelT[:, None]
# print(f'new shape is {labelT.shape}')

# use scikitlearn to split the data
train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelT, test_size=.2)

# convert them into PyTorch Datasets 
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

# Translate into dataloader objects
batchsize = 32
train_loader = DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

'''
# Check sizes of data batches 
for x, y in train_loader:
    # Remeber to set drop_last=True
    print(x.shape, y.shape)
'''

class HARModel(nn.Module):
    # New class inherited from the nn.module
    def __init__(self):
        super().__init__() # `super()` allows python to access function from nn.module (parent class)
        # input layer
        self.input = nn.Linear(6,16)

        # hidden layer
        self.hidden1=nn.Linear(16, 32)

        # output layer
        self.output = nn.Linear(32,1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        return self.output(x)



numepochs=500

def trainTheModel():
    lossfun=nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(HARnet.parameters(),lr=0.1)

    # initialize loss
    losses=torch.zeros(numepochs)
    trainAcc=[]
    testAcc=[]

    # loop over epochs
    for epochi in range(numepochs):
        HARnet.train()  # sets the model to training mode

        # loop over training data batches
        batchAcc=[]
        batchLoss=[]

        for x, y in train_loader:
            # forward pass and losses
            yHat = HARnet(x)
            loss = lossfun(yHat,y)

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # losses from this batch
            batchLoss.append(loss.item())

            # compute training accuracy for this batch
            batchAcc.append( 100*torch.mean(((yHat>0) == y).float()).item() )
        
        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append( np.mean(batchAcc) )
        # end of batch loop
        losses[epochi] = np.mean(batchLoss)

        # test accuracy
        HARnet.eval()
        x_1,y = next(iter(test_loader)) # extract X, y from test dataloader
        with torch.no_grad(): # deactivates auto grad
            yHat=HARnet(x_1)

        testAcc.append( 100*torch.mean(((yHat>0) == y).float()).item())

    # function output
    return trainAcc, testAcc, losses

HARnet=HARModel()
trainAcc,testAcc,losses = trainTheModel()
# print(losses)

fig, ax = plt.subplots(1,2,figsize=(6,3))

ax[0].plot(trainAcc)
ax[1].plot(testAcc)

plt.show()

        