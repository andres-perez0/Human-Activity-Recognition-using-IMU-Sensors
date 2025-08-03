/*
  Arduino Wireless Communication Code
  Reciever Code
  Inspired and Learnt from Dejan Nedelkovski and Arduino Forums
  Link: https://howtomechatronics.com/tutorials/arduino/arduino-wireless-communication-nrf24l01-tutorial/
  Link: https://projecthub.arduino.cc/tmekinyan/how-to-use-the-nrf24l01-module-with-arduino-813957 
*/

#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

#define CE_PIN 7 
#define CSN_PIN 8 

RF24 radio(CE_PIN, CSN_PIN);

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
  Serial.begin(9600);

  radio.begin();
  radio.setAutoAck(false); //(bool) Append ACK packet from the receiving radio back to the transmitting radio
  radio.setDataRate(RF24_2MBPS); // (RF24_250KBPS|RF24_1MBPS|RF24_2MBPS) 
  radio.setPALevel(RF24_PA_MIN);
  radio.setPayloadSize(sizeof(payLoad)); 
  radio.openReadingPipe(0, address);
  radio.startListening();

  Serial.println("reading for recieving");
}
void loop() {
  
  if (radio.available() > 0) {
    read_Data();
  } else {
    Serial.println("waiting");
  }
}

void read_Data() {
  /*  
    Sends formatted data in a comma-seperated string
    Format: "accX,accY,accZ,gyroX,gyroY,gyroZ"
  */
  radio.read(&payLoad, sizeof(payLoad)); 

  Serial.print(payLoad.accX);
  Serial.print(",");
  Serial.print(payLoad.accZ);
  Serial.print(",");
  Serial.print(payLoad.accZ);
  Serial.print(",");
  Serial.print(payLoad.gyroX);
  Serial.print(",");
  Serial.print(payLoad.gyroZ);
  Serial.print(",");
  Serial.println(payLoad.gyroY);
}