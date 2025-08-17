from dataCtrl import dataCtrl

# for DP modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# for number - crunching
import numpy as np

# for dataset management
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt

# for file mangement
import time

class HARModel(nn.Module):
    # New class inherited from the nn.module
    def __init__(self, input_features, num_classes):
        super(HARModel, self).__init__() # `super()` allows python to access function from nn.module (parent class)
        
        self.conv1=nn.Conv1d(
            in_channels=input_features,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * window_size, 128)
        self.fc2 = nn.Linear(128, num_classes) # num_classes = 2 (sitting, walking)

    def forward(self, x):
        x=x.permute(0,2,1)

        # Forward pass through the convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        
        # Flatten the output for the fully connected layers
        x = self.flatten(x)
        
        # Forward pass through the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def trainTheModel(lr):
    lossfun=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(HARnet.parameters(),lr=lr)

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
            _, predicted = torch.max(yHat.data, 1)
            batchAcc.append( 100 * (predicted == y).float().mean().item() )
        
        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append( np.mean(batchAcc) )
        # end of batch loop
        losses[epochi] = np.mean(batchLoss)


        # Correct way to calculate test accuracy
        HARnet.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                outputs_test = HARnet(x_test)
                _, predicted_test = torch.max(outputs_test.data, 1)
                total_samples += y_test.size(0)
                correct_predictions += (predicted_test == y_test).sum().item()
        
        testAcc.append(100 * correct_predictions / total_samples)
    
    # function output
    return trainAcc, testAcc, losses

DataProcessor = dataCtrl()

data = DataProcessor.initialize_data()
window_size_linspace=np.linspace(100,600,12)

trainAcc_list=[]
testAcc_list=[]
totalTime_list=[]

for i in range(len(window_size_linspace)):

    window_size=int(window_size_linspace[i])
    overlap_percentage=0.3

    train_data, test_data, train_labels, test_labels = DataProcessor.preprocess_and_spilt(data, window_size, overlap_percentage,0.2)

    # convert them into PyTorch Datasets 
    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)

    # Translate into dataloader objects
    batchsize = 16
    train_loader = DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batchsize)

    input_features = 6
    num_classes = 2
    numepochs=500
    learning_rate=0.005

    startTime=time.time()
    HARnet=HARModel(input_features, num_classes)
    trainAcc,testAcc,losses = trainTheModel(0.005)

    endTime=time.time()

    trainAcc_list.append(trainAcc[-1])
    testAcc_list.append(testAcc[-1])
    totalTime_list.append(endTime-startTime)


fig, ax = plt.subplots(1,2,figsize=(10,5))

# Plotting Accuracy (%) vs. Learning Rate
ax[0].plot(window_size_linspace, trainAcc_list, 'b-', label='Training Accuracy')
ax[0].plot(window_size_linspace, testAcc_list, 'r-', label='Testing Accuracy')
ax[0].set_title('Accuracy vs. Window Size')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_xlabel('Window Size (#)')
ax[0].legend()
ax[0].grid(True)

# Plotting window_size vs. Learning Rate
ax[1].plot(window_size_linspace, totalTime_list, 'g-')
ax[1].set_title('Training Time vs. Window Size')
ax[1].set_ylabel('Time (s)')
ax[1].set_xlabel('Window Size(#)')
ax[1].grid(True)

plt.show()

print(f'window_size : {window_size}')
print(f'overlap % : {overlap_percentage}')
print(f'batch_size : {batchsize}')
print(f'numepochs : {numepochs}')

print(f'final training : {trainAcc_list[-1]}')
print(f'final testing : {testAcc_list[-1]}')
print(f'final total time : {totalTime_list[-1]}')

max=testAcc_list[0]
max_i = 0
for i, val in enumerate(testAcc_list):
    if val > max:
        max   = val
        max_i = i

print('below is the information of the model the performed best in testing ')
print(f'train acc : {trainAcc_list[max_i]}')
print(f'best test acc : {testAcc_list[max_i]}')
print(f'total time : {totalTime_list[max_i]}')
print(f'window size : {window_size_linspace[max_i]}')

'''
parameterized_experiments_window_size.png
window_size : 600
overlap % : 0.3
batch_size : 16
numepochs : 500
final training : 100.0
final testing : 75.0
final total time : 33.10504674911499
below is the information of the model the performed best in testing
train acc : 100.0
best test acc : 100.0
total time : 96.97022795677185
window size : 190.9090909090909
'''

