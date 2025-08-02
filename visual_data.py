import matplotlib.pyplot as plt
import pandas as pd

fileName="IMU_Data/mpu9250_data13.csv"
fig,ax=plt.subplots(2,3,figsize=(8,8))

csvDataFrame=pd.read_csv(fileName)

incompleteRowMask=csvDataFrame.isnull().any(axis=1)

# print('Incomplete Rows:') # print(csvDataFrame[incompleteRowMask])

csvFiltered=csvDataFrame[incompleteRowMask != True]
csvLength=len(csvFiltered)

# data=np.zeros((csvLength, 6),dtype=float)
# print(csvFiltered)

data=csvFiltered.iloc[:, 2:8]

# print(type(data))
# print(data.loc[:, 'gyroX'])

index=0
graphColors=['r','g','b','r','g','b']
graphLabels=['accX','accY','accZ','gyroX','gyroY','gyroZ']

for ix in range(2):
    for iy in range(3):
        ax[ix,iy].plot(data.loc[:,graphLabels[index]], graphColors[index])
        ax[ix,iy].set_title(graphLabels[index])
        index+=1

plt.tight_layout()
plt.show()