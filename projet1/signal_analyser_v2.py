import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import json

files = [
    # #"SDS824X_HD_Binary_C1_2",
    # "SDS824X_HD_Binary_C1_3",
    # "SDS824X_HD_Binary_C1_4",
    # #"SDS824X_HD_Binary_C1_5",
    # #"SDS824X_HD_Binary_C1_6",
    # "SDS824X_HD_Binary_C1_7",
    # #"SDS824X_HD_Binary_C1_8",
    # "SDS824X_HD_Binary_C1_9",
    # #"SDS824X_HD_Binary_C1_10",
    # "SDS824X_HD_Binary_C1_11",
    # "SDS824X_HD_Binary_C1_12",
    # "SDS824X_HD_Binary_C1_13",
    # "SDS824X_HD_Binary_C1_14",
    # "SDS824X_HD_Binary_C1_15",
    # # "SDS824X_HD_Binary_C1_16",
    # "SDS824X_HD_Binary_C1_17",
    # #"SDS824X_HD_Binary_C1_18",
    # #"SDS824X_HD_Binary_C1_19",
    # "SDS824X_HD_Binary_C1_20",
    # #"SDS824X_HD_Binary_C1_21",
    # #"SDS824X_HD_Binary_C1_22",
    # "SDS824X_HD_Binary_C1_23",
    # "SDS824X_HD_Binary_C1_24",
    # #"SDS824X_HD_Binary_C1_25",
    "SDS824X_HD_Binary_C1_26",
    "SDS824X_HD_Binary_C1_27",
]


def analyze_pulses(voltages, sample_rate, low_threshold=0.06, high_threshold=0.06, window_size=15, min_samples=100):
    """
    Improved version with Temporal Hysteresis to prevent "flickering" pulses.
    
    :param min_samples: Minimum number of samples the signal must stay 
                        above/below threshold to trigger a state change.
    """
    
    # 1. Noise Reduction
    # Increased default window_size to 15 for better smoothing
    kernel = np.ones(window_size) / window_size
    smoothed_v = np.convolve(voltages, kernel, mode='same')
    
    is_pulse = False
    pulse_indices = []
    
    current_start = None
    high_streak = 0
    low_streak = 0
    
    for i, v in enumerate(smoothed_v):
        if not is_pulse:
            if v > high_threshold:
                high_streak += 1
                if high_streak >= min_samples:
                    is_pulse = True
                    # Set start to when the streak began
                    current_start = i - min_samples 
                    low_streak = 0
            else:
                high_streak = 0
        else:
            if v < low_threshold:
                low_streak += 1
                if low_streak >= min_samples:
                    is_pulse = False
                    # End the pulse where the drop-off began
                    pulse_indices.append((current_start, i - min_samples))
                    high_streak = 0
            else:
                low_streak = 0

    # 3. Convert to Time (List of Dictionaries)
    pulse_data = []
    for start_idx, end_idx in pulse_indices:
        # Prevent negative indices or invalid durations
        start_idx = max(0, start_idx)
        pulse_data.append({
            "start_time": start_idx / sample_rate,
            "end_time": end_idx / sample_rate,
            "duration": (end_idx - start_idx) / sample_rate
        })
        
    return pulse_data, smoothed_v

def analyze_biphasic_pulses(voltages, sample_rate, window_size=21,
                              min_pulse_duration=1e-7, max_pulse_duration=5e-4,
                              min_amplitude=0.05):
    """
    Detect biphasic pulses where each event starts on a local dip and peaks on a local rise.

    Added amplitude/duration filtering to reject high-frequency noise and spurious pulses.

    :param min_pulse_duration: minimum duration in seconds for a valid pulse.
    :param max_pulse_duration: maximum duration in seconds for a valid pulse.
    :param min_amplitude: minimum peak-to-valley amplitude to qualify as a pulse.
    """
    # Smooth the signal to reduce high-frequency noise
    kernel = np.ones(window_size) / window_size
    smoothed_v = np.convolve(voltages, kernel, mode='same')

    # Compute first derivative and locate extrema
    dv = np.diff(smoothed_v)
    if len(dv) < 2:
        return [], smoothed_v

    local_mins = np.where((dv[:-1] < 0) & (dv[1:] >= 0))[0] + 1
    local_maxs = np.where((dv[:-1] > 0) & (dv[1:] <= 0))[0] + 1

    pulse_data = []
    if local_mins.size > 1 and local_maxs.size > 0:
        left_mins = local_mins[:-1]
        right_mins = local_mins[1:]

        # Find max candidates in between successive minima via vectorized search
        max_start_pos = np.searchsorted(local_maxs, left_mins, side='right')
        max_end_pos = np.searchsorted(local_maxs, right_mins, side='left')

        valid = max_start_pos < max_end_pos
        if np.any(valid):
            candidate_starts = left_mins[valid]
            candidate_ends = local_maxs[max_end_pos[valid] - 1]

            for s, e in zip(candidate_starts, candidate_ends):
                if e <= s:
                    continue

                duration = (e - s) / sample_rate
                if duration < min_pulse_duration or duration > max_pulse_duration:
                    continue

                local_amp = smoothed_v[e] - smoothed_v[s]
                if local_amp < min_amplitude:
                    continue

                # Avoid pulses embedded in very jittery regions by requiring a real dip and rise.
                if smoothed_v[s] >= smoothed_v[e]:
                    continue

                pulse_data.append({
                    "start_time": float(s) / sample_rate,
                    "end_time": float(e) / sample_rate,
                    "duration": float(duration),
                    "amplitude": float(local_amp)
                })

    return pulse_data, smoothed_v

def analyze_negative_pulses(voltages, sample_rate, bridge_samples=1000, min_drop_samples=50):
    # --- STAGE 1: IN-PLACE MORPHOLOGICAL CLOSING ---
    # We still use this to clean the "fuzz" on the high parts
    bridged_v = maximum_filter1d(voltages, size=bridge_samples)
    bridged_v = minimum_filter1d(bridged_v, size=bridge_samples)

    # --- STAGE 2: DYNAMIC THRESHOLDING ---
    v_top = np.percentile(bridged_v, 98) 
    v_min = np.min(bridged_v)           
    v_range = v_top - v_min
    
    # Hysteresis for Positive-Going (High) detection
    # 'rise_thresh': Signal must get back near the top to start the 'High' pulse
    # 'fall_thresh': Signal must drop significantly to end the 'High' pulse
    rise_thresh = v_top - (v_range * 0.08) 
    fall_thresh = v_top - (v_range * 0.12) 

    # --- STAGE 3: DETECTION LOOP (High Logic) ---
    pulse_indices = []
    is_high_pulse = False
    current_start = 0
    drop_streak = 0 
    
    for i, v in enumerate(bridged_v):
        if not is_high_pulse:
            # START of High Pulse: Signal rises above threshold
            if v > rise_thresh:
                is_high_pulse = True
                current_start = i
                drop_streak = 0
        else:
            # END of High Pulse: Signal drops below threshold
            if v < fall_thresh:
                drop_streak += 1
                if drop_streak >= min_drop_samples:
                    is_high_pulse = False
                    # End the pulse at the moment the drop started
                    pulse_indices.append((current_start, i - drop_streak))
            else:
                drop_streak = 0 

    # Handle case where the signal ends while still in a 'High' state
    if is_high_pulse:
        pulse_indices.append((current_start, len(bridged_v) - 1))

    # --- STAGE 4: OUTPUT FORMATTING ---
    pulse_data = []
    for start_idx, end_idx in pulse_indices:
        if end_idx > start_idx:
            pulse_data.append({
                "start_time": start_idx / sample_rate,
                "end_time": end_idx / sample_rate,
                "duration": (end_idx - start_idx) / sample_rate
            })
        
    return pulse_data, bridged_v

def analyze_positive_pulses(voltages, sample_rate, bridge_samples=1000, min_fall_samples=30, rise_thresh=0.1, fall_thresh=0.104):
    # --- STAGE 1: IN-PLACE MORPHOLOGICAL OPENING ---
    # Clean noise at low values and preserve rising activity
    bridged_v = minimum_filter1d(voltages, size=bridge_samples)
    bridged_v = maximum_filter1d(bridged_v, size=bridge_samples)

    # --- STAGE 2: DYNAMIC THRESHOLDING ---
    v_bottom = np.percentile(bridged_v, 6)
    v_max = np.percentile(bridged_v, 94)
    v_range = v_max - v_bottom

    # Hysteresis for Positive-Going (Rise) detection
    # 'rise_thresh': Signal must climb above this to start a rising pulse
    # 'fall_thresh': Signal must drop below this to end the rising pulse
    #rise_thresh = v_bottom + (v_range * rising_affinity)
    #fall_thresh = v_max - (v_range * falling_affinity)
    rise_thresh = rise_thresh
    fall_thresh = fall_thresh

    print("_" * 100)
    print(v_bottom)
    print(v_max)
    print(rise_thresh)
    print(fall_thresh)
    print("_" * 100)

    # --- STAGE 3: DETECTION LOOP (Rise Logic) ---
    pulse_indices = []
    is_rise_pulse = False
    current_start = 0
    fall_streak = 0

    for i, v in enumerate(bridged_v):
        if not is_rise_pulse:
            # START of Rise Pulse: Signal rises above threshold
            if v > rise_thresh:
                is_rise_pulse = True
                current_start = i
                fall_streak = 0
        else:
            # END of Rise Pulse: Signal falls below lower threshold
            if v < fall_thresh:
                fall_streak += 1
                if fall_streak >= min_fall_samples:
                    is_rise_pulse = False
                    pulse_indices.append((current_start, i - fall_streak))
            else:
                fall_streak = 0

    # Handle case where the signal ends while still in a 'Rise' state
    if is_rise_pulse:
        pulse_indices.append((current_start, len(bridged_v) - 1))

    # --- STAGE 4: OUTPUT FORMATTING ---
    pulse_data = []
    for start_idx, end_idx in pulse_indices:
        if end_idx > start_idx:
            pulse_data.append({
                "start_time": start_idx / sample_rate,
                "end_time": end_idx / sample_rate,
                "duration": (end_idx - start_idx) / sample_rate
            })

    return pulse_data, bridged_v


def analyze_positive_pulses_with_derivative(voltages, sample_rate, rise_deriv_threshold=None, fall_deriv_threshold=None, min_pulse_samples=500000, max_pulse_samples=None, refractory_samples=5000):
    """Minimal derivative-based pulse detection.

    - start on a positive derivative spike (likely 1-2 samples only)
    - end on a negative derivative spike
    - simple output, minimal state, tuned to hard edges
    - refractory_samples: number of samples to ignore after a pulse ends
    """
    bridged_v = minimum_filter1d(voltages, size=1000)
    bridged_v = maximum_filter1d(bridged_v, size=1000)
    dv = np.gradient(voltages) * float(sample_rate)
    v_bottom = np.percentile(bridged_v, 3)

    # Wait for a low baseline before starting detection
    baseline_indices = np.where(bridged_v <= v_bottom)[0]
    if baseline_indices.size == 0:
        return [], bridged_v

    detection_start_idx = int(baseline_indices[0] + 500)
    detection_start_idx = min(detection_start_idx, len(dv))

    abs_med = np.median(np.abs(dv))
    abs_std = np.std(dv)
    edge_floor = abs_med + 3.0 * abs_std

    if rise_deriv_threshold is None:
        rise_deriv_threshold = max(edge_floor, 1e-8)
    if fall_deriv_threshold is None:
        fall_deriv_threshold = min(-edge_floor, -1e-8)

    pulse_indices = []
    in_pulse = False
    current_start = None
    next_valid_start = detection_start_idx

    for i in range(detection_start_idx, len(dv)):
        d = dv[i]
        if i < next_valid_start:
            continue

        if not in_pulse:
            if d >= rise_deriv_threshold:
                in_pulse = True
                current_start = i
        else:
            if d <= fall_deriv_threshold:
                if current_start is not None and (i - current_start) >= min_pulse_samples:
                    pulse_indices.append((current_start, i))
                    in_pulse = False
                    next_valid_start = i + refractory_samples
                    current_start = None

        if in_pulse and current_start is not None and max_pulse_samples is not None:
            if i - current_start > max_pulse_samples:
                pulse_indices.append((current_start, i))
                in_pulse = False
                next_valid_start = i + refractory_samples
                current_start = None

    if in_pulse and current_start is not None and (len(bridged_v) - current_start) >= min_pulse_samples:
        pulse_indices.append((current_start, len(bridged_v) - 1))

    pulse_data = []
    for start_idx, end_idx in pulse_indices:
        if end_idx > start_idx:
            pulse_data.append({
                "start_time": (start_idx / sample_rate) - 0.000016,
                "end_time": end_idx / sample_rate - 0.000006,
                "duration": ((end_idx - start_idx) / sample_rate)
            })

    return pulse_data, bridged_v


def fuse_pulses_nearest_neighbor(robust_pulses, precision_pulses):
    """
    For each pulse in robust_pulses, finds the single closest start and 
    the single closest end timestamp in precision_pulses, but only if the
    precision times are at or after the robust times.
    
    :param robust_pulses: The 'stable' pulses (Anchors).
    :param precision_pulses: The 'high-precision' pulses (Lookup Table).
    """
    # 1. Extract all possible start and end times from the precision data
    # We flatten these into arrays so we can do fast math
    p_starts = np.array([p['start_time'] for p in precision_pulses])
    p_ends = np.array([p['end_time'] for p in precision_pulses])
    
    if len(p_starts) == 0 or len(p_ends) == 0:
        print("Warning: No precision pulses found to match against.")
        return []

    fused_results = []

    for anchor in robust_pulses:
        a_start = anchor['start_time']
        a_end = anchor['end_time']

        # 2. Pick only precision starts that are >= robust start
        valid_starts = p_starts[p_starts >= a_start]
        if valid_starts.size == 0:
            continue
        idx_start = np.abs(valid_starts - a_start).argmin()
        best_start = valid_starts[idx_start]

        # 3. Pick only precision ends that are >= robust end
        valid_ends = p_ends[p_ends >= a_end]
        if valid_ends.size == 0:
            continue
        idx_end = np.abs(valid_ends - a_end).argmin()
        best_end = valid_ends[idx_end]

        # 4. Ensure we preserve pulse ordering (end after start)
        if best_end <= best_start:
            later_ends = valid_ends[valid_ends > best_start]
            if later_ends.size == 0:
                continue
            best_end = later_ends[0]

        fused_results.append({
            "start_time": best_start,
            "end_time": best_end,
            "duration": best_end - best_start,
            "original_start": a_start,
            "shift_ms": (best_start - a_start) * 1000  # How much precision changed it
        })

    return fused_results

def patch_voltage_dips(volt, sample_rate, end_time=0.05, start_time=0, threshold_factor=0.2):
    """
    Finds min/max values and replaces 'low' values with the 'max' value 
    within a specific time range.
    
    :param volt: 1D numpy array of voltages.
    :param start_time: Start of the range to fix (in seconds).
    :param end_time: End of the range to fix (in seconds).
    :param sample_rate: e.g., 1000000000 (1 GSa/s).
    :param threshold_factor: Values within this % of the minimum are considered 'dips'.
    """
    # 1. Calculate indices from time
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(volt), end_idx)
    
    # 2. Find Minimum and Maximum in the entire signal (or the segment)
    v_min = np.min(volt)
    v_max = np.max(volt)
    
    print(f"Global Min: {v_min:.3f}V, Global Max: {v_max:.3f}V")
    
    # 3. Create a copy to modify
    fixed_volt = volt.copy()
    
    # 4. Define what counts as a "min value" to be replaced
    # Since noise prevents values from being exactly the same, 
    # we use a threshold (e.g., bottom 20% of the signal swing).
    gap_threshold = v_min + (threshold_factor * (v_max - v_min))
    
    # 5. Extract the range and replace values
    segment = fixed_volt[start_idx:end_idx]
    
    # Boolean mask: Find all points in this range that are 'low'
    dips_mask = (segment <= gap_threshold)
    
    # Set those points to the maximum
    segment[dips_mask] = v_max
    
    # Put the fixed segment back into the array
    fixed_volt[start_idx:end_idx] = segment
    
    return fixed_volt

def patch_voltage_peaks(volt, sample_rate, end_time=0.05, start_time=0, threshold_factor=0.2):
    """
    Finds max/min values and replaces 'high' values with the 'min' value 
    within a specific time range (inverse of patch_voltage_dips).

    :param volt: 1D numpy array of voltages.
    :param start_time: Start of the range to fix (in seconds).
    :param end_time: End of the range to fix (in seconds).
    :param sample_rate: e.g., 1000000000 (1 GSa/s).
    :param threshold_factor: Values within this % of the maximum are considered 'peaks'.
    """
    # 1. Calculate indices from time
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)

    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(volt), end_idx)

    # 2. Find Minimum and Maximum in the entire signal (or the segment)
    v_min = np.min(volt)
    v_max = np.max(volt)

    print(f"Global Min: {v_min:.3f}V, Global Max: {v_max:.3f}V")

    # 3. Create a copy to modify
    fixed_volt = volt.copy()

    # 4. Define what counts as a "max value" to be replaced
    # Since noise prevents values from being exactly the same,
    # we use a threshold (e.g., top 20% of the signal swing).
    peak_threshold = v_max - (threshold_factor * (v_max - v_min))

    # 5. Extract the range and replace values
    segment = fixed_volt[start_idx:end_idx]

    # Boolean mask: Find all points in this range that are 'high'
    peaks_mask = (segment >= peak_threshold)

    # Set those points to the minimum
    segment[peaks_mask] = v_min

    # Put the fixed segment back into the array
    fixed_volt[start_idx:end_idx] = segment

    return fixed_volt

def plot_complete_with_highlights(time, volt, pulse_results, smoothed_signal=None, file_name="Waveform"):
    """
    Plots the entire voltage waveform and overlays highlights for every detected pulse.
    """
    # 1. Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # 2. Plot the COMPLETE voltage line first (Blue)
    # Using a slightly thinner line and better alpha for high-density data
    #ax.plot(time, volt, label='Voltage Signal', color='royalblue', linewidth=0.7, alpha=0.8)
    plt.plot(time, volt, color='blue', alpha=0.3, label='Raw')
    if smoothed_signal is not None:
        plt.plot(time, smoothed_signal, color='black', label='Processed (Bridged)')

    # 3. Add the Highlights (Red Shading)
    # We loop through results but stay on the SAME 'ax'
    has_labeled = False
    for pulse in pulse_results:
        # axvspan creates a vertical rectangle from start_time to end_time
        ax.axvspan(pulse['start_time'], pulse['end_time'], 
                   color='red', alpha=0.3, 
                   label='Detected Pulses' if not has_labeled else "")
        has_labeled = True

    # 4. Add Threshold lines (Optional, helps verify detection logic)
    # plt.axhline(y=high_threshold, color='green', linestyle='--', alpha=0.5, label='High Trigger')

    # 5. Final Formatting
    ax.set_title(f"Complete Waveform Analysis: {file_name}", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    # Optimization for large datasets: 
    # This prevents the x-axis from cutting off data
    ax.set_xlim(time[0], time[-1])

    plt.tight_layout()
    plt.show()

def plot_pulse_comparison(time1, volt1, pulses1, time2, volt2, pulses2, title="Pulse Detection Comparison"):
    """
    Plots two signals and highlights the detected pulses for comparison.
    Uses decimation for plotting to save memory.
    """
    # Decimation factor for plotting only (every 100th point)
    # Adjust this if the plot looks too 'thin' or 'chunky'
    step = 100
    step1 = 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # --- TOP PLOT: Dataset 1 ---
    ax1.plot(time1[::step1], volt1[::step1], color='royalblue', alpha=0.7, label='Signal 1 (Raw)')
    for i, p in enumerate(pulses1):
        # Draw a shaded span for the pulse duration
        ax1.axvspan(p['start_time'], p['end_time'], color='orange', alpha=0.3)
        # Add a marker at the start for precision check
        if i == 0: # Label only first one for legend
            ax1.axvline(p['start_time'], color='red', linestyle='--', alpha=0.5, label='Pulse Detect')
        else:
            ax1.axvline(p['start_time'], color='red', linestyle='--', alpha=0.5)

    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Dataset 1: Robust Detection")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- BOTTOM PLOT: Dataset 2 ---
    ax2.plot(time2[::step], volt2[::step], color='seagreen', alpha=0.7, label='Signal 2 (Patched)')
    for p in pulses2:
        ax2.axvspan(p['start_time'], p['end_time'], color='gold', alpha=0.3)
        ax2.axvline(p['start_time'], color='darkred', linestyle='--', alpha=0.5)

    ax2.set_ylabel("Voltage (V)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Dataset 2: Negative/High Pulse Detection")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analyze_pulse_deltas(robust_pulses, precision_pulses):
    """
    Calculates the timing offset (error/difference) between two sets of pulse detections.
    
    Returns a list of dictionaries with deltas and a summary statistics dictionary.
    """
    # 1. Prepare precision lookup arrays
    p_starts = np.array([p['start_time'] for p in precision_pulses])
    p_ends = np.array([p['end_time'] for p in precision_pulses])
    
    if len(p_starts) == 0:
        return [], {}

    delta_results = []
    
    for i, anchor in enumerate(robust_pulses):
        a_s = anchor['start_time']
        a_e = anchor['end_time']

        # 2. Find closest matches in the precision set
        idx_s = np.abs(p_starts - a_s).argmin()
        idx_e = np.abs(p_ends - a_e).argmin()
        
        prec_s = p_starts[idx_s]
        prec_e = p_ends[idx_e]

        # 3. Calculate Deltas (Precision - Robust)
        # Positive value = Precision detected it LATER
        # Negative value = Precision detected it EARLIER
        start_delta = prec_s - a_s
        end_delta = prec_e - a_e
        
        delta_results.append({
            "pulse_index": i,
            "start_delta_ms": start_delta,
            "end_delta_ms": end_delta,
            "distance1": start_delta * 3e8,
            "distance2": end_delta * 3e8,
            "mean_distance": (start_delta * 3e8 + start_delta * 3e8) / 2
        })

    # 4. Generate Summary Stats
    df = pd.DataFrame(delta_results)
    stats = {
        "avg_start_delta_ms": df["start_delta_ms"].mean(),
        "max_start_delta_ms": df["start_delta_ms"].abs().max(),
        "min_start_delta_ms": df["start_delta_ms"].abs().min(),
        "avg_end_delta_ms": df["end_delta_ms"].mean(),
        "max_end_delta_ms": df["start_delta_ms"].abs().max(),
        "min_end_delta_ms": df["start_delta_ms"].abs().min(),
        "avg_distance1": df["distance1"].mean(),
        "avg_distance2": df["distance2"].mean(),
        "avg_distance": df["mean_distance"].mean(),
    }
    print(stats["avg_distance"])
    print(max(stats["max_start_delta_ms"], stats["max_end_delta_ms"]))
    print(min(stats["min_start_delta_ms"], stats["min_end_delta_ms"]))
    return delta_results, stats

if __name__ == '__main__':
    for file_name in files:
        file_name2 = file_name.replace("_C1_", "_C2_")
        csv_file2 = f"data/jour 5/treatus_fleubuis/{file_name2}.csv"
        fs = 1000000000
        index = int(file_name.split("_")[-1])


        # csv_file = f"data/jour 5/treatus_fleubuis/{file_name}.csv"
        # data = pd.read_csv(csv_file)
        # volt = data["voltage (V)"].to_numpy() * -1
        # time = np.arange(len(volt)) * 1e-9
        # pulse_results1, smoothed_signal1 = analyze_positive_pulses_with_derivative(volt, fs)
        # plot_complete_with_highlights(time[::1000], volt[::1000], pulse_results1, file_name=file_name)
        data2 = pd.read_csv(csv_file2)
        volt2 = data2["voltage (V)"].to_numpy()
        time2 = np.arange(len(volt2)) * 1e-9

        rise_thresh=-0.0275
        fall_thresh=-0.0270

        if index in [4, 26]:
            volt_fixed2 = volt2
            pulse_results2, smoothed_signal2 = analyze_positive_pulses(volt2, fs)
            #pulse_results2_aligned = fuse_pulses_nearest_neighbor(pulse_results1, pulse_results2)
        elif index in [3, 7, 9]:
            volt_fixed2 = patch_voltage_dips(volt2, start_time=0.0, end_time=0.05, sample_rate=fs)
            pulse_results2, smoothed_signal2 = analyze_positive_pulses(volt_fixed2, fs, rise_thresh=rise_thresh, fall_thresh=fall_thresh)
            #pulse_results2_aligned = fuse_pulses_nearest_neighbor(pulse_results1, pulse_results2)
        elif index in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]:
            volt_fixed2 = patch_voltage_peaks(volt2, start_time=0.0, end_time=0.05, sample_rate=fs)
            pulse_results2, smoothed_signal2 = analyze_positive_pulses(volt_fixed2, fs, rise_thresh=rise_thresh, fall_thresh=fall_thresh)
            #pulse_results2_aligned = fuse_pulses_nearest_neighbor(pulse_results1, pulse_results2)
        elif index in [2, 6, 8, 5]:
            volt_fixed2 = volt2
            pulse_results2, smoothed_signal2 = analyze_positive_pulses(volt2, fs)
            #pulse_results2_aligned = fuse_pulses_nearest_neighbor(pulse_results1, pulse_results2)
        elif index in [10]:
            volt_fixed2 = patch_voltage_dips(volt2, start_time=0.0, end_time=0.05, sample_rate=fs)
            pulse_results2, smoothed_signal2 = analyze_negative_pulses(volt2, fs)
            #pulse_results2_aligned = fuse_pulses_nearest_neighbor(pulse_results1, pulse_results2)

        #plot_complete_with_highlights(time2[::1000], volt_fixed2[::1000], pulse_results2, file_name=file_name)
        plot_complete_with_highlights(time2[::1000], volt2[::1000], pulse_results2, smoothed_signal=smoothed_signal2[::1000], file_name=file_name)

        #plot_pulse_comparison(time, smoothed_signal1, pulse_results1, time2, smoothed_signal2, pulse_results2)
        # plot_pulse_comparison(time, smoothed_signal1, pulse_results1, time2, smoothed_signal2, pulse_results2_aligned)

        # delta_results, stats = analyze_pulse_deltas(pulse_results1, pulse_results2_aligned)
        # save_dict = {"delta_results": delta_results, "stats": stats}
        # with open(f"data/jour 5/treatus_fleubuis/treated_data/{file_name}.json", 'w', encoding='utf-8') as f:
        #     json.dump(save_dict, f, indent=4)