import struct
import numpy as np
import pandas as pd

files = [
    "SDS824X_HD_Binary_C1_2",
    "SDS824X_HD_Binary_C1_3",
    "SDS824X_HD_Binary_C1_4",
    "SDS824X_HD_Binary_C1_5",
    "SDS824X_HD_Binary_C1_6",
    "SDS824X_HD_Binary_C1_7",
    "SDS824X_HD_Binary_C1_8",
    "SDS824X_HD_Binary_C1_9",
    "SDS824X_HD_Binary_C1_10",
    "SDS824X_HD_Binary_C1_11",
    "SDS824X_HD_Binary_C1_12",
    "SDS824X_HD_Binary_C1_13",
    "SDS824X_HD_Binary_C1_14",
    "SDS824X_HD_Binary_C1_15",
    "SDS824X_HD_Binary_C1_16",
    "SDS824X_HD_Binary_C1_17",
    "SDS824X_HD_Binary_C1_18",
    "SDS824X_HD_Binary_C1_19",
    "SDS824X_HD_Binary_C1_20",
    "SDS824X_HD_Binary_C1_21",
    "SDS824X_HD_Binary_C1_22",
    "SDS824X_HD_Binary_C1_23",
    "SDS824X_HD_Binary_C1_24",
    "SDS824X_HD_Binary_C1_25",
    "SDS824X_HD_Binary_C1_26",
    "SDS824X_HD_Binary_C1_27",
    "SDS824X_HD_Binary_C2_2",
    "SDS824X_HD_Binary_C2_3",
    "SDS824X_HD_Binary_C2_4",
    "SDS824X_HD_Binary_C2_5",
    "SDS824X_HD_Binary_C2_6",
    "SDS824X_HD_Binary_C2_7",
    "SDS824X_HD_Binary_C2_8",
    "SDS824X_HD_Binary_C2_9",
    "SDS824X_HD_Binary_C2_10",
    "SDS824X_HD_Binary_C2_11",
    "SDS824X_HD_Binary_C2_12",
    "SDS824X_HD_Binary_C2_13",
    "SDS824X_HD_Binary_C2_14",
    "SDS824X_HD_Binary_C2_15",
    "SDS824X_HD_Binary_C2_16",
    "SDS824X_HD_Binary_C2_17",
    "SDS824X_HD_Binary_C2_18",
    "SDS824X_HD_Binary_C2_19",
    "SDS824X_HD_Binary_C2_20",
    "SDS824X_HD_Binary_C2_21",
    "SDS824X_HD_Binary_C2_22",
    "SDS824X_HD_Binary_C2_23",
    "SDS824X_HD_Binary_C2_24",
    "SDS824X_HD_Binary_C2_25",
    "SDS824X_HD_Binary_C2_26",
    "SDS824X_HD_Binary_C2_27",
]

def get_v4_data_with_unit(buffer, start_addr):
    """Parses 'Data With Unit' (Table 2/3) for V4.0 format."""
    # Value is a 64-bit float [cite: 259]
    value = struct.unpack('<d', buffer[start_addr : start_addr + 8])[0]
    # Magnitude index is a 32-bit integer [cite: 259]
    mag_idx = struct.unpack('<i', buffer[start_addr + 8 : start_addr + 12])[0]
    
    # Magnitudes (Table 3): 7=milli, 8=IU/Unit, 9=kilo [cite: 261]
    magnitudes = {5: 1e-9, 6: 1e-6, 7: 1e-3, 8: 1, 9: 1e3, 10: 1e6}
    return value * magnitudes.get(mag_idx, 1)

def extract_v4_data(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(4096)

        wave_len = struct.unpack('<I', header[0x1ec:0x1f0])[0]
        data_width = struct.unpack('<B', header[0x264:0x265])[0] # 0=8-bit, 1=16-bit
        data_offset = struct.unpack('<I', header[0x04:0x08])[0]
        
        active_channels = []
        for i in range(8):
            if struct.unpack('<i', header[0x08 + (i * 4) : 0x0c + (i * 4)])[0] == 1:
                active_channels.append(i + 1)
        
        f.seek(data_offset)
        # 16-bit HD data must be read as SIGNED (h) to prevent inversion
        dtype = np.int16 if data_width == 1 else np.uint8
        raw_bytes = f.read(wave_len * len(active_channels) * (2 if data_width == 1 else 1))
        all_raw_data = np.frombuffer(raw_bytes, dtype=dtype)
        
        all_voltages = []
        for idx, ch_num in enumerate(active_channels):
            ch_raw = all_raw_data[idx * wave_len : (idx + 1) * wave_len].astype(float)
            
            # Retrieve scaling factors [cite: 247, 250]
            vdiv = get_v4_data_with_unit(header, 0x18 + (ch_num - 1) * 0x28)
            voffset = get_v4_data_with_unit(header, 0xb8 + (ch_num - 1) * 0x28)
            
            cpd_addr = 0x270 + (ch_num - 1) * 4
            code_per_div = struct.unpack('<i', header[cpd_addr : cpd_addr + 4])[0]
            
            # --- THE FIX ---
            if data_width == 1: # 16-bit (HD)
                # For signed int16, center_code is 0. 
                # We SUBTRACT the offset if it was appearing inverted.
                voltages = (ch_raw * (vdiv / code_per_div)) - voffset
            else: # 8-bit
                # Standard unsigned 8-bit conversion [cite: 238, 455]
                voltages = (ch_raw - 128) * (vdiv / code_per_div) - voffset
            
            all_voltages.append(voltages)
            
        return np.array(all_voltages)

if __name__ == '__main__':
    for file_name in files:
        bin_file = f"data/jour 5/raw_data/{file_name}.bin"
        csv_file = f"data/jour 5/treatus_fleubuis/{file_name}.csv"
        voltages = extract_v4_data(bin_file)[0]

        df = pd.DataFrame({
            "voltage (V)": voltages
        })
        df.to_csv(csv_file, index=False)