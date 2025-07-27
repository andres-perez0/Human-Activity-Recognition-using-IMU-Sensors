import serial
import time
import datetime
import csv

class SerialCtrl():
    def __init__(self):
        self.COM='COM6'
        self.BD_RATE=9600
        self.TIME_OUT=0.1

        self.file_index=0
        self.filename=f"mpu9250_data{self.file_index}.csv"

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
        print('please enter the acitivity you intend to measure for a minute: ')
        self.activity_label = input('> ').strip().lower()        

    def StartStream(self):
        # Implement Threading Logic Here

        start_time=time.time()
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

if __name__=="__main__":
    serial_control=SerialCtrl()
    serial_control.SerialOpen()
    serial_control.SelectActivity()
    
    try:
        serial_control.StartStream()
    except KeyboardInterrupt:
        print("user terminated")
        serial_control.SerialClose()

    serial_control.SerialClose()