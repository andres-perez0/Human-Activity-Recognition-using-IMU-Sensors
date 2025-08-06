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

DataProcessor = dataCtrl()

data = DataProcessor.initialize_data()
window_size=400
overlap_percentage=0.3
train_data, test_data, train_labels, test_labels = DataProcessor.preprocess_and_spilt(data, window_size, overlap_percentage,0.2)

# convert them into PyTorch Datasets 
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

# Translate into dataloader objects
batchsize = 16
train_loader = DataLoader(train_data, batch_size=batchsize,shuffle=True,drop_last=True)
test_loader = DataLoader(test_data, batch_size=batchsize)

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

        # input layer
        # self.input = nn.Linear(6, 16)

        # hidden layer 8 minutes and 23 seconds
        # self.hidden1=nn.Linear(16, 32)
        # self.hidden2=nn.Linear(32, 32)

        # # output layer
        # self.output = nn.Linear(32, 1)

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
        # x = F.relu(self.input(x))
        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # return self.output(x)

numepochs=500

def trainTheModel(lr):
    lossfun=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(HARnet.parameters(),lr=lr)

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

        ''' # test accuracy
        HARnet.eval()
        x,y = next(iter(test_loader)) # extract X, y from test dataloader
        with torch.no_grad(): # deactivates auto grad
            yHat=HARnet(x)
        testAcc.append( 100*torch.mean(((yHat>0) == y).float()).item())'''

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

input_features = 6
num_classes = 2
# learning_rate=0.001
lr_linspace=np.linspace(0.001,0.1,20)

trainAcc_list=[]
testAcc_list=[]
totalTime_list=[]

for i in range(len(lr_linspace)): 
    startTime=time.time()
    HARnet=HARModel(input_features, num_classes)
    trainAcc,testAcc,losses = trainTheModel(lr_linspace[i])
    endTime=time.time()

    trainAcc_list.append(trainAcc[-1])
    testAcc_list.append(testAcc[-1])
    totalTime_list.append(endTime-startTime)
    # print(f"your model train for {int((endTime-startTime) // 60)} minutes and {int((endTime-startTime) % 60)} seconds")
    
# startTime=time.time()
# HARnet=HARModel(input_features, num_classes)
# trainAcc,testAcc,losses = trainTheModel(learning_rate)

# endTime=time.time()
# print(f"your model train for {int((endTime-startTime) // 60)} minutes and {int((endTime-startTime) % 60)} seconds")

fig, ax = plt.subplots(1,2,figsize=(10,5))

# Plotting Accuracy (%) vs. Learning Rate
ax[0].plot(lr_linspace, trainAcc_list, 'b-', label='Training Accuracy')
ax[0].plot(lr_linspace, testAcc_list, 'r-', label='Testing Accuracy')
ax[0].set_title('Accuracy vs. Learning Rate')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_xlabel('Learning Rate')
ax[0].legend()
ax[0].grid(True)

# Plotting Training Time (s) vs. Learning Rate
ax[1].plot(lr_linspace, totalTime_list, 'g-')
ax[1].set_title('Training Time vs. Learning Rate')
ax[1].set_ylabel('Time (s)')
ax[1].set_xlabel('Learning Rate')
ax[1].grid(True)

plt.show()

'''
11
your model train for 0 minutes and 13 seconds
the final trading accuracy is 100.0
the final testing accuracy is 79.16666666666667
learning_rate 0.001
12
your model train for 0 minutes and 31 seconds
the final trading accuracy is 100.0
the final testing accuracy is 87.5
learning_rate 0.001
'''

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
