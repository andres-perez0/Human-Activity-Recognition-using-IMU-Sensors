import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np

import scipy.stats as stats

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
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
batchsize = 64
train_loader = DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

# # Check sizes of data batches 
# for x, y in train_loader:
#     # Remeber to set drop_last=True
#     print(x.shape, y.shape)