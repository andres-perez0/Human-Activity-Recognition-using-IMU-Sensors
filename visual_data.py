import matplotlib.pyplot as plt
import pandas as pd

class visualData():
    def __init__(self,fileName):
        self.fileName = fileName
        self.fig, self.ax = plt.subplots(2,3,figsize=(9,9))

    def initializeDataFrame(self):
        # self.csvDataFrame = pd.read_csv(self.fileName,header=[0,5],on_bad_lines='skip')
        # self.incompleteRowMask = self.csvDataFrame.isnull().any(axis=1)

        # csvFiltered=self.csvDataFrame[self.incompleteRowMask != True]
        # self.csvLength=len(csvFiltered)

        # self.data=csvFiltered.iloc[:, 2:8]
        self.data=pd.read_csv(self.fileName,header=[0,5],on_bad_lines='skip').iloc[:,2:8]
    
    def additionalInformation(self):
        print('Incomplete Rows:') 
        print(self.csvDataFrame[self.incompleteRowMask])
    
    def graphData(self):
        index=0
        graphColors=['r','g','b','r','g','b']
        graphLabels=['accX','accY','accZ','gyroX','gyroY','gyroZ']

        for ix in range(2):
            for iy in range(3):
                self.ax[ix,iy].plot(self.data.loc[:,graphLabels[index]], graphColors[index])
                self.ax[ix,iy].set_title(graphLabels[index])
                index+=1
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__": 
    fileName="IMU_Data/mpu9250_data0.csv"
    visualDataInstance=visualData(fileName)

    visualDataInstance.initializeDataFrame()
    visualDataInstance.graphData()
