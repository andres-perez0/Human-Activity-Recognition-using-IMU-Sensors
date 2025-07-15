/*
  IMU Data Sender By Andres Perez
*/
#include <MPU9250.h>
#include <Wire.h>

MPU9250 mpu;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Wire.begin();
  delay(2000);

  if (!mpu.setup(0x68)) {
    while (1) {
      Serial.println("MPU connection failed. Please check your connection!");
      delay(5000);
    }
  }

  Serial.println("Arduino Ready");
}

void loop() {
  /*
    Don't forget the mpu.update()!
  */
  if (mpu.update()) {
    send_data();
  }
  delay(100); // 100 Hz; 50 ms for 20 Hz
}

void send_data() {
  /*  
    Sends formatted data in a comma-seperated string
    Format: "accX,accY,accZ,gyroX,gyroY,gyroZ"
  */
  Serial.print(mpu.getAccX());
  Serial.print(",");
  Serial.print(mpu.getAccY());
  Serial.print(",");
  Serial.print(mpu.getAccZ());
  Serial.print(",");
  Serial.print(mpu.getGyroX());
  Serial.print(",");
  Serial.print(mpu.getGyroY());
  Serial.print(",");
  Serial.println(mpu.getGyroZ());
}