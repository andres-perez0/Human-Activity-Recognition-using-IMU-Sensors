import matplotlib.pyplot as plt
import numpy as np
import csv

fig,ax=plt.subplots(2,3,figsize=(8,8))

count=0
with open('IMU_Data/mpu9250_data4.csv', newline='') as csvfile:
    rowreader = csv.reader(csvfile, quotechar='|')
    header=next(rowreader,None)
    
    if header:
        print(f"Header: {header}")
    for row in rowreader:
        count+=1

data=np.zeros((count, 6),dtype=float)
index=0

with open('IMU_Data/mpu9250_data4.csv', newline='') as csvfile:
    rowreader = csv.reader(csvfile, quotechar='|')
    header=next(rowreader,None)
    
    if header:
        print(f"Header: {header}")
    for row in rowreader:
        data[index,0]=float(row[2])
        data[index,1]=float(row[3])
        data[index,2]=float(row[4])
        data[index,3]=float(row[5])
        data[index,4]=float(row[6])
        data[index,5]=float(row[7])
        index+=1

index=0
graphColors=['r','g','b','r','g','b']
graphTitles=['accX','accY','accZ','gyroX','gyroY','gyroZ']

for ix in range(2):
    for iy in range(3):
        ax[ix,iy].plot(data[:,index], graphColors[index])
        ax[ix,iy].set_title(graphTitles[index])
        index+=1

plt.tight_layout()
plt.show()