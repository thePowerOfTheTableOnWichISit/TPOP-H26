import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

values = [
    {"index": 3, "distance": 0.56},
    {"index": 4, "distance": 0.90},
    {"index": 7, "distance": 1.8},
    {"index": 9, "distance": 2.4},
    {"index": 11, "distance": 3},
    {"index": 12, "distance": 3.3},
    {"index": 13, "distance": 3.6},
    {"index": 14, "distance": 3.9},
    {"index": 15, "distance": 4.2},
    {"index": 16, "distance": 4.2},
    {"index": 17, "distance": 4.5},
    {"index": 20, "distance": 5.4},
    {"index": 23, "distance": 7.0},
    {"index": 24, "distance": 8.0},
]

# map index -> theoretical distance
values_map = {int(item["index"]): float(item["distance"]) for item in values}


def plot_avg_distance_by_theoretical_distance(folder_path):
    folder = Path(folder_path)
    pattern = re.compile(r"SDS824X_HD_Binary_C1_(\d+)\.(csv|json)$", re.IGNORECASE)

    rows = []

    for fp in sorted(folder.glob("SDS824X_HD_Binary_C1_*.csv")) + sorted(folder.glob("SDS824X_HD_Binary_C1_*.json")):
        m = pattern.search(fp.name)
        if not m:
            continue

        idx = int(m.group(1))
        theoretical_distance = values_map.get(idx)
        avg_distance = None
        values_tostd = []

        text = fp.read_text(encoding="utf-8", errors="ignore").strip()

        # If JSON format, load it
        if fp.suffix.lower() == ".json" or text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict):
                peaks = payload.get("delta_results", [])
                
                for peak in peaks:
                    values_tostd.append(peak["mean_distance"] / 2)

                avg_distance = payload.get("stats", {}).get("avg_distance")
                # Fallback index-distance in file content (for completeness)
                for key in ("array", "data", "entries", "distances"):
                    arr = payload.get(key)
                    if isinstance(arr, list) and theoretical_distance is None:
                        for item in arr:
                            if isinstance(item, dict) and item.get("index") == idx:
                                theoretical_distance = item.get("distance")
                                break

        if avg_distance is None:
            # CSV mode: try table read. The file might hold a JSON-like record too.
            try:
                df_csv = pd.read_csv(fp)
                if "avg_distance" in df_csv.columns:
                    avg_distance = float(df_csv["avg_distance"].iloc[0])

                if theoretical_distance is None and {"index", "distance"}.issubset(df_csv.columns):
                    match = df_csv.loc[df_csv["index"] == idx]
                    if not match.empty:
                        theoretical_distance = float(match["distance"].iloc[0])
            except Exception:
                pass

        # Skip if no avg_distance (not interpretable)
        if avg_distance is None:
            print(f"Skipping {fp.name}: avg_distance not found")
            continue

        # If theoretical distance still not resolved, use index as fallback or skip
        if theoretical_distance is None:
            theoretical_distance = float(idx)
            print(f"Warn: No predefined distance for index {idx}; using index as x.")

        rows.append((idx, theoretical_distance, float(avg_distance) / 2, np.std(values_tostd)))

    if not rows:
        raise RuntimeError("No valid avg_distance results found in folder")

    df_out = pd.DataFrame(rows, columns=["index", "theoretical_distance", "avg_distance", "std"]).sort_values("theoretical_distance")

    plt.figure(figsize=(7, 5))
    plt.scatter(df_out["theoretical_distance"], df_out["avg_distance"], marker="o", s=150)
    plt.errorbar(df_out["theoretical_distance"], df_out["avg_distance"], yerr=df_out["std"], linestyle="None")

    plt.tick_params(axis='both', labelsize=30) 
    plt.xlabel("Distance entre laser et capteur B (m)", fontsize=50)
    plt.ylabel("Distance mesurée (m)", fontsize=50)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return df_out


if __name__ == "__main__":
    base_folder = Path(r"D:\uni\TPOP---Projet-1\data\jour 5\treatus_fleubuis\treated_data")
    print("Scanning folder:", base_folder)
    df_result = plot_avg_distance_by_theoretical_distance(base_folder)
    print(df_result.to_string(index=False))
