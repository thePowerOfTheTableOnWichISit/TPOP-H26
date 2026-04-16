import numpy as np
import csv
import struct
import pandas as pd


files_C1 = [
    "SDS824X_HD_Binary_C1_2",
    "SDS824X_HD_Binary_C1_3",
    "SDS824X_HD_Binary_C1_4",
]

files_C2 = [
    "SDS824X_HD_Binary_C2_2",
    "SDS824X_HD_Binary_C2_3",
    "SDS824X_HD_Binary_C2_4",
]

header_size = 512
def generate_csv_from_bin(raw, vertical_scale, vertical_offset, time_step, csv_file):
    data = raw[0x1470:]
    samples = np.frombuffer(data, dtype=np.uint16)

    voltage = (samples - 128) * vertical_scale / 25 + vertical_offset
    time = np.arange(len(voltage)) * time_step

    df = pd.DataFrame({
        "time (s)": time,
        "voltage (V)": voltage
    })
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    # file exported from oscilloscope
    for file_name in files_C2:
        bin_file = f"data/jour 5/raw_data/{file_name}.bin"
        csv_file = f"data/jour 5/treated data/{file_name}.csv"

        vertical_scale = 0.01
        vertical_offset = -0.0118
        time_step = 0.5e-9

        with open(bin_file, "rb") as f:
            raw = f.read()
        generate_csv_from_bin(raw, vertical_scale, vertical_offset, time_step, csv_file)
    
    for file_name in files_C1:
        bin_file = f"data/jour 5/raw_data/{file_name}.bin"
        csv_file = f"data/jour 5/treated data/{file_name}.csv"

        vertical_scale = 5
        vertical_offset = 0
        time_step = 1e-9

        with open(bin_file, "rb") as f:
            raw = f.read()
        generate_csv_from_bin(raw, vertical_scale, vertical_offset, time_step, csv_file)