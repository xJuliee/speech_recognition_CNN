// Code written by Juliette Gelderland and Ana Antohi
// Communication between a Python CNN, ESP8266, ATmega2560 and the RP6 Robot
// CNN passes on classified commands via WiFi -> ESP8622 passes on commands via UART -> ATmega2560 uses I2C to pass on movement routines to RP6

#include "Rp6.h" // Library from B3nzchr3ur: https://github.com/b3nzchr3ur/arduino-rp6-library

void initializeESP8266() {
  delay(1000);
  Serial3.println("AT");                                    // Test communication
  delay(1000);
  Serial3.println("AT+CWJAP=\"YAJ81JVQ9\",\"Pinkpink\"");   // Connect to WiFi
  delay(5000);                                              // Wait some time to establish a connection
  Serial3.println("AT+CWMODE=1");                           // Set to station mode
  delay(500);
  Serial3.println("AT+CIPMUX=1");                           // Enable connection
  delay(500);
  Serial3.println("AT+CIPSERVER=1,5000");                   // Start TCP server on port 5000
  delay(1000);
  Serial3.println("AT+CIFSR");                               // Get the EP8266's IP address
  delay(1000);
  Serial.println("ESP8266 WiFi + TCP server initialized");
}

void setup() {
  Serial.begin(9600);       // Debugging via USB Serial Monitor
  Serial3.begin(115200);    // ESP8266+ATmega2560 UART communication
  Rp6.begin();
  initializeESP8266();
  Serial.println("Waiting for TCP commands via WiFi...");
}

void loop() {
  if (Serial3.available()) {
    String command = Serial3.readStringUntil('\n');
    command.trim(); // Remove any extra whitespace/newlines

    // Check if it starts with +IPD -> extract the actual command
    if (command.startsWith("+IPD")) {
      int colonIndex = command.indexOf(':');
      if (colonIndex != -1 && colonIndex + 1 < command.length()) {
        command = command.substring(colonIndex + 1);
      }
    }

    Serial.println("Received from Python: " + command);

    if (command == "left") {
      Rp6.rotate(50, RP6_LEFT, 90);
      Serial.println("Turning left");
      delay(500);
    }
    else if (command == "right") {
      Rp6.rotate(50, RP6_RIGHT, 90);
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
      Serial.println("Unknown command. Try: left, right, forward, back, stop");
    }
  }
}

