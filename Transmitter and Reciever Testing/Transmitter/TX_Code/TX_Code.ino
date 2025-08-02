/*
  Arduino Wireless Communication Code
  Transmitter Code
  Inspired and Learnt from Dejan Nedelkovski and Arduino Forums
  Link: https://howtomechatronics.com/tutorials/arduino/arduino-wireless-communication-nrf24l01-tutorial/
  Link: https://projecthub.arduino.cc/tmekinyan/how-to-use-the-nrf24l01-module-with-arduino-813957 
*/

#include <MPU9250.h>
#include <nRF24L01.h>
#include <RF24.h>
#include <SPI.h>
#include <Wire.h>

#define CE_PIN 7
#define CSN_PIN 8

RF24 radio(CE_PIN, CSN_PIN); 
MPU9250 mpu;

const byte address[6] = "00001";

struct payLoad {
  // NR24L01 buffer limit is 32 byte (max struct size)
  // floats are 4 bytes; 4 * 6 features => 24 bytes of data
  float accX;
  float accY;
  float accZ;
  float gyroX;
  float gyroY;
  float gyroZ;
};

payLoad payLoad;

void setup() {
  // Serial Functions commented out because we're tramsitting a packet
  // Not reading from the serial line on the transmitter end
  // Serial.begin(9600);
  radio.begin();
  radio.setAutoAck(false); // (bool) | Append ACK packet from the receiving radio back to the transmitting radio  
  radio.setDataRate(RF24_2MBPS); //(RF24_250KBPS|RF24_1MBPS|RF24_2MBPS)
  // Greater level = more consumption = longer distance 
  // Power Amplifier - Primary function is to amplify the Radio Frequency signal before its transmitted
  radio.setPALevel(RF24_PA_MAX); // (RF24_PA_MIN|RF24_PA_LOW|RF24_PA_HIGH|RF24_PA_MAX) 
  radio.setPayloadSize(sizeof(payLoad)); // Default value is the maximum 32 bytes; Reminder gets replaced by 0s
  radio.openWritingPipe(address); // Act as transmitter
  radio.stopListening();

  Wire.begin();

  if (!mpu.setup(0x68)) {
    while (1) {
      // Serial.println("MPU connection failed. Please check your connection");
      delay(5000);
    }
  }
  // Serial.println("Arduino Ready");
}

void loop() {
  if (mpu.update()) {
    send_data();
  }
  delay(100);
}

void send_data() {
  /*  
    Loads the payLoad Struct with the accelerometer and gyroscope information
    Writes to the location in the setUp() function
    Format: {accX,accY,accZ,gyroX,gyroY,gyroZ}
  */
  payLoad.accX=mpu.getAccX();
  payLoad.accY=mpu.getAccY();
  payLoad.accZ=mpu.getAccZ();
  payLoad.gyroX=mpu.getGyroX();
  payLoad.gyroY=mpu.getGyroY();
  payLoad.gyroZ=mpu.getGyroZ();

  // Testing Print statements
  // Serial.print("Testing AccX: ");
  // Serial.println(payLoad.accX); 

  radio.write(&payLoad,sizeof(payLoad));
}

