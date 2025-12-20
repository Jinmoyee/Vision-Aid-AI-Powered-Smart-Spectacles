import serial
import time

# Replace '/dev/ttyUSB0' with the correct port (e.g., 'COM3' on Windows)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # Wait for the connection to establish

print("Reading distance from ultrasonic sensor...")

try:
    while True:
        if ser.in_waiting > 0:
            distance = ser.readline().decode('utf-8').strip()
            print(distance)
except KeyboardInterrupt:
    print("\nExiting...")
    ser.close()
