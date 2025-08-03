from read_data import SerialCtrl
from visual_data import visualData

if __name__=="__main__":
    
    # serial_control=SerialCtrl()
    # fileName=serial_control.updateFileName()

    # serial_control.SerialOpen()

    # while True:
    #     try:
    #         serial_control.SelectActivity()
    #         serial_control.StartStream()
    #     except KeyboardInterrupt:
    #         print("user terminated")
    #         serial_control.SerialClose()
    #         break

    visual_Data1=visualData('IMU_Data\mpu9250_data8.csv')
    visual_Data1.initializeDataFrame()
    visual_Data1.graphData()

    # serial_control.SerialClose()