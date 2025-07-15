import serial
import datetime
import csv

COM='COM6'
BD_RATE=9600
TIME_OUT=0.1
FILENAME="data.csv"

global uart
uart=True

def my_serial():
    global uart
    ser = serial.Serial(port=COM,baudrate=BD_RATE,timeout=TIME_OUT)

    with open(FILENAME, "w", newline='') as f: 
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'accX', 'accY', 'accZ','gyroX', 'gyroY', 'gyroZ'])
        while uart:
            data=ser.readline()
            if len(data) > 0:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ")
                    # print(timestamp, end='')

                    sensor_data=data.decode('utf-8').strip()
                    information = sensor_data.split(',')
                    information = [float(item) for item in information]
                    # print(*information)
                    row_to_write = [timestamp] + information 

                    writer.writerow(row_to_write)
                except:
                    pass
    ser.close()

try:     
    my_serial()
except KeyboardInterrupt:
    print("User Terminated the program")
    uart=False
