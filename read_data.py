from visual_data import visualData
import serial
import time
import os
import datetime
import csv

class SerialCtrl():
    def __init__(self):
        self.COM='COM3'
        self.BD_RATE=9600
        self.TIME_OUT=0.1

        self.file_index=0
        self.folder_name="IMU_Data"
        self.filename=f"mpu9250_data{self.file_index}.csv"

    def checkIfFileExist(self, currFilename):
        ''' Checks if the existence of the file before running the next data collection'''
        file_path = os.path.join(self.folder_name, currFilename)

        if os.path.exists(file_path):
            print(f"{file_path} already exists")
            return True
        else: 
            print(f"{file_path} does not exist")
            return False

    def updateFileName(self):
        while self.checkIfFileExist(self.filename):
            currentFile=os.path.join(self.folder_name, self.filename)

            self.file_index += 1
            self.filename=f"mpu9250_data{self.file_index}.csv"
        
        print(f"Your new file is {self.filename}")
        return currentFile
        

    def SerialOpen(self):
        try: 
            self.ser.is_open
        except:
            self.ser=serial.Serial(port=self.COM,baudrate=self.BD_RATE,timeout=self.TIME_OUT)
        
        try: 
            if self.ser.is_open:
                print("Already Open")
                self.ser.status=True
            else: 
                self.ser=serial.Serial(port=self.COM,baudrate=self.BD_RATE,timeout=self.TIME_OUT)
                print("Opening Serial")
                self.ser.open()
                self.ser.status=True
        except:
            self.ser.status=False

    def SerialClose(self):
        try:
            self.ser.is_open
            self.ser.close()
            self.ser.status=False
        except:
            print("Already closed")
            self.ser.status = False

    def SelectActivity(self):
        print(f"current file name is {self.filename}. Remember type Ctrl + C to close serial connection.")
        print('please enter the acitivity you intend to measure for a minute: ')
        self.activity_label = input('> ').strip().lower()        

    def StartStream(self):
        print("please get into position you have 5 seconds")
        
        for i in range(5):
            print(f"starts in {5-(i+1)}")
            time.sleep(1) 

        # Clear the input buffer to get rid of any old data
        self.ser.flushInput()

        start_time=time.time()
        self.filename=os.path.join(self.folder_name,self.filename)
        with open(self.filename, "w", newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'activity_label', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ'])
            
            while time.time() - start_time < 60:             
                data=self.ser.readline()

                if len(data) > 0:
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ")
                        # print(timestamp, end='')
                        sensor_data=data.decode('utf-8').strip()

                        information = sensor_data.split(',')
                        information = [float(item) for item in information]
                        # print(*information)
                        row_to_write = [timestamp] + [self.activity_label] + information 

                        writer.writerow(row_to_write)
                    except:
                        pass

        print(f"{self.filename} complete! :)")

        self.file_index += 1
        self.filename=f"mpu9250_data{self.file_index}.csv"

