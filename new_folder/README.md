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
- The Arduino Mega2560 will act as the I2C Master, sending movement commands based on classified speech input.
