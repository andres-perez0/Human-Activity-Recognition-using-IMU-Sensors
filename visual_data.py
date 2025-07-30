import matplotlib.pyplot as plt
import numpy as np
import csv

fig,ax=plt.subplots(2,2,figsize=(7,7))

count=0
with open('IMU_Data/mpu9250_data1.csv', newline='') as csvfile:
    rowreader = csv.reader(csvfile, quotechar='|')
    header=next(rowreader,None)
    
    if header:
        print(f"Header: {header}")

    for row in rowreader:
        count+=1
        # print(row[0])
# print(count)

data=np.zeros((count, 6),dtype=float)
index=0

with open('IMU_Data/mpu9250_data1.csv', newline='') as csvfile:
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

print(data[:,0])

ax[0,0].plot(data[:,0],'r',label='accX')
ax[0,1].plot(data[:,1],'g',label='accY')
ax[1,0].plot(data[:,2],'b',label='accZ')

plt.legend()
plt.show()