# # Counting the number of rows in a csv file
# import pandas as pd

# df=pd.read_csv('IMU_Data_2\mpu9250_data15.csv')
# csvLength = len(df)

# print(f"the df is {csvLength} rows.\nthe df shape is {df.shape}")

# # Checking for missing values in rows
# missingValuesPercolumn=df.isnull().sum()

# print("Missing values per column:")
# print(missingValuesPercolumn)

# incompleteRowMask=df.isnull().any(axis=1)

# incompleteRowCount=incompleteRowMask.sum()

# print('\n Incomplete Rows')
# print(df[incompleteRowMask])

# df_filtered = df [ incompleteRowMask != True]

# print(df_filtered)

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

'''
# Check sizes of data batches 
for x, y in train_loader:
    # Remeber to set drop_last=True
    print(x.shape, y.shape)
'''


## Sliding Windows
# Examples from https://medium.com/@machinelearningclub/sliding-window-algorithms-for-real-time-data-processing-2012f00d07d7

# 1. Network Traffic Analysis
'''
Network traffic analysis is critical for ensuring the security and performance of network systems. Sliding window algorithms can aid in detecting anomalies in network traffic patterns
'''
# traffic_data=[100, 120, 150, 200, 300, 50, 400, 250, 180, 350]

# def detect_anomalies(traffic_data, window_size):
#     anomalies=[]

#     for i in range(len(traffic_data) - window_size + 1):
#         window=traffic_data[i: i+window_size]

#         if sum(window) > 400:
#             anomalies.append(window)
        
#     return anomalies

# anomalies=detect_anomalies(traffic_data, 3)
# print("anomalies in network traffic data: ")
# [print(anomaly) for anomaly in anomalies]

# 2. Data Stream Processing
'''
In scenarios where data arrives continously in a stream, such as sensor data
or financial market data, computing moving averages over a sliding window is a
common requirement
'''
# data_stream = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# def moving_averages(data_stream,window_size):
#     moving_average=[]
    
#     for i in range(len(data_stream) - window_size + 1):
#         window=data_stream[i:i+window_size]
#         moving_average.append(sum(window)/window_size)
#     return moving_average

# averages = moving_averages(data_stream,3)
# [print(avg) for avg in averages]

## use z-score to find anoamalies

data_stream = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

import pandas as pd

data = pd.read_csv('combine_mpu9250.csv', sep=',')

'''
3. iterate through the dataframe
- your loop should iterate throught the selected sensor data, not the entire data frame
- range function should be based on the number or rows in your selected data from the beginning in rend

4. Extract the Windows
- Inside the loop, for each iteration i, you'll slive the dataframe to get a window of data.
The slice will be from i to i + window_size. This will create a small data frame from all the sensor data

5. Handle the activity label
- The activity_label is a single label for a window, not a series of labels. to handl ethis, you need to decided how to 
assign an activity label to each window

- common approach: take the most frequeny activity_label within the window. If all rows in a window
are from the same activity, this is straightforward. if not you need to perform a simple
aggregation (mode()) on acativity_labe

- Create a list to store the segmented data, each item in the list will be a window which could be either a small
dataframe along with its corresponding label

- return the list of windows 
'''

import numpy as np
import pandas as pd 

def sliding_window(data, window_size, overlap_percentage):
    '''
        data: pandas DataFrame
        window_size: int
        overlap_percentage: float (0 < i < 1)
        Gemini walked through this function with me
    '''
    # Data columns to be segmented
    sensor_cols=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

    # Step size based on window size and overlap
    step_size = int(window_size * (1-overlap_percentage))

    if step_size <= 0:
        raise ValueError("calculated step size is zero or negative")
    
    windows_with_labels=[]

    # Iterate through the df in steps
    for i in range(0, len(data) - window_size + 1, step_size):
        window       = data.iloc[i:i+window_size][sensor_cols]
        window_label = data.iloc[i:i+window_size]['activity_label'].mode()

        # windows_with_labels.append((window, window_label))

    return windows_with_labels

window_list = sliding_window(data, 250, 0.3)

# [print(x.shape,y.shape) for x,y in window_list]
# print(len(window_list))
print(window_list[0])