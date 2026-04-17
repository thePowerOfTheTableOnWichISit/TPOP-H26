import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
file_name = "SDS824X_HD_Binary_C1_3"
file_name2 = "SDS824X_HD_Binary_C2_3"
csv_file = f"data/jour 5/treatus_fleubuis/{file_name}.csv"
csv_file2 = f"data/jour 5/treatus_fleubuis/{file_name2}.csv"


data = pd.read_csv(csv_file)
data2 = pd.read_csv(csv_file2)
data2.columns = ["time (s)", "voltage (V)"]

volt = data["voltage (V)"].to_numpy()
time = np.arange(len(volt)) * 1e-9


volt2 = data["voltage (V)"].to_numpy()
time2 = np.arange(len(volt2)) * 1e-9
# Plot

time = time[::1000]
volt = volt[::1000]
time2 = time2[::1000]
volt2 = volt2[::1000] * 200

plt.figure()
plt.plot(time, volt)
plt.plot(time2, volt2)

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Time")
plt.grid(True)


plt.show()