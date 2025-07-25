import serial
import datetime
import csv

class SerialCtrl():
    def __init__(self):
        self.COM='COM6'
        self.BD_RATE=9600
        self.TIME_OUT=0.1
        self.FILENAME="data.csv"
        self.uart=True

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
        self.activity_label = input('please enter the acitivity you intend to measure: ').strip()        

    def StartStream(self):
        # Implement Threading Logic Here
        with open("data.csv", "w", newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'activity_label', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ'])
            while self.uart:
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

if __name__=="__main__":
    serial_control=SerialCtrl()
    serial_control.SerialOpen()
    serial_control.SelectActivity()
    try: 
        serial_control.StartStream()
    except KeyboardInterrupt:
        print("User ended streaming")
        
    serial_control.SerialClose()

