import serial
import csv
import time

PORT = 'COM3'        # Replace with your Arduino port (Windows example)
BAUDRATE = 115200    # Must match Serial.begin() in Arduino
CSV_FILENAME = 'data.csv'

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

print("Reading serial data... Press Ctrl+C to stop.")

with open(CSV_FILENAME, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['time_us', 'detValue'])
    try:
        while True:
            line = ser.readline().decode('utf-8').strip() 
            if line:
                values = line.split(',')
                if len(values) == 2:
                    csv_writer.writerow(values)
                    print(values)
    except KeyboardInterrupt:
        print("\nStopped by user.")

ser.close()