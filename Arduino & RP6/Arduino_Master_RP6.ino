// Code written by Juliette Gelderland and Ana Antohi
// Communication between a Python CNN, ATmega2560 and the RP6 Robot
// CNN passes on classified commands via UART/USB -> ATmega2560 uses I2C to pass on movement routines to RP6

#include "Rp6.h" // Library from B3nzchr3ur: https://github.com/b3nzchr3ur/arduino-rp6-library

void setup()
{
  Serial.begin(9600);
  Rp6.begin();
  Serial.println("Type 'left', 'right', 'forward', 'back', or 'stop'");
}

const int someDelay = 200;
const uint8_t someSpeed = 50;

void loop()
{
  if (Serial.available())
  {
    String command = Serial.readStringUntil('\n');
    command.trim(); // remove any trailing newline or whitespace

    if (command == "left") {
      Rp6.rotate(50, RP6_LEFT, 90); // Speed 50, rotate left by 90 degrees
      Serial.println("Turning left");
      delay(500);
    }
    else if (command == "right") {
      Rp6.rotate(50, RP6_RIGHT, 90); // Speed 50, rotate right by 90 degrees
      Serial.println("Turning right");
      delay(500);
    }
    else if (command == "forward") {
      Rp6.move(80, RP6_FORWARD, 1000);
      Serial.println("Moving forward");
      delay(500);
    }
    else if (command == "back") {
      Rp6.move(80, RP6_BACKWARD, 1000);
      Serial.println("Moving backward");
      delay(500);
    }
    else if (command == "stop") {
      Rp6.stop();
      Serial.println("Stopped");
      delay(500);
    }
    else {
      Serial.println("Unknown command. Try: left, right, forward, backward, stop");
    }
  }
}
