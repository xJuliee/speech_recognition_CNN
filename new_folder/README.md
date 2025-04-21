# RP6 Slave Code & Library
The communication between the Arduino Mega2560 and the RP6 Robot is handled via I2C using the following custom library: https://github.com/b3nzchr3ur/arduino-rp6-library

# Set-up:
Upload Slave Code to RP6 Robot
- Flash the following code to the RP6 using its development environment: https://github.com/b3nzchr3ur/rp6/tree/master/RP6Examples_20120725f/RP6BASE_EXAMPLES/

Install RP6 Arduino Library
- Download the ZIP of the RP6 Arduino library from the link above.
- In the Arduino IDE, go to Sketch > Include Library > Add .ZIP Library
- Select the downloaded ZIP file to install it.

Once this setup is complete:
- The RP6 Robot will act as the I2C Slave.
- The Arduino Mega2560 will act as the I2C Master.

# Arduino Code:
Choose the appropriate Arduino sketch based on your hardware setup:
  No ESP8266 or other WiFi module?
- Use Arduino_Master_RP6.ino
  Using ESP8266 WiFi module?
- Use Arduino_Master_RP6_WiFi.ino

Upload the selected .ino file to your Arduino Mega2560 using the Arduino IDE.
Modify the code as needed:
- Set your WiFi IP address (for WiFi version).
- Adjust movement routines if they differ from the default.

Connect your Arduino to the RP6 Robot following the schematic below:
![{F7B62DCD-AD5C-472C-B946-A30AF7A1D306}](https://github.com/user-attachments/assets/d0d3734e-b2c6-4f3c-8058-7454b052e9a7)

Note: If you are not using an ESP8266 WiFi module, establish the connection between your laptop and Arduino via USB/UART instead of WiFi.
