import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import re

import numpy as np
import pandas as pd
from klupus2 import *

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

def _extract_numeric_signal_from_json(obj):
    """Try to find a suitable numeric signal array in JSON content."""
    if isinstance(obj, list):
        # If list of numbers
        if len(obj) > 0 and all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj, dtype=float)
        # If list of dicts with numeric fields
        if len(obj) > 0 and isinstance(obj[0], dict):
            # pick first numeric column in dicts
            keys = list(obj[0].keys())
            for k in keys:
                values = [item.get(k) for item in obj if isinstance(item, dict) and k in item]
                if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                    return np.array(values, dtype=float)
    elif isinstance(obj, dict):
        # if there is an obvious signal key
        candidates = ["signal", "values", "data", "trace", "voltage", "v"]
        for k in candidates:
            v = obj.get(k)
            if isinstance(v, list) and len(v) > 1 and all(isinstance(x, (int, float)) for x in v):
                return np.array(v, dtype=float)
        # fallback to dict values arrays
        for v in obj.values():
            if isinstance(v, list) and len(v) > 1 and all(isinstance(x, (int, float)) for x in v):
                return np.array(v, dtype=float)
    return None


def _get_signal_from_file(fp: Path):
    """Try to construct a 1D numeric signal from file content."""
    try:
        if fp.suffix.lower() == ".json":
            payload = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
            arr = _extract_numeric_signal_from_json(payload)
            if arr is not None:
                return arr
            raise ValueError("cannot find numeric vector in JSON file")

        df = pd.read_csv(fp, header=0)

        # If there is exactly one column, assume it is potential.
        if df.shape[1] == 1:
            col = df.columns[0]
            data = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(data) < 2:
                raise ValueError("no numeric points in single-column CSV")
            return np.asarray(data, dtype=float)

        # If 2+ columns exist, prefer second column as potential
        if df.shape[1] >= 2:
            second_col = df.columns[1]
            data = pd.to_numeric(df[second_col], errors="coerce").dropna().values
            if len(data) < 2:
                raise ValueError("no numeric points in second column")
            return np.asarray(data, dtype=float)

        raise ValueError("unexpected CSV format")

    except Exception as e:
        raise e


def compute_stats(signal: np.ndarray):
    """Compute SNR, SEM, and SINAD from an array signal."""
    if signal.size < 2:
        raise ValueError("signal too short")

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    sem_val = float(std_val / math.sqrt(signal.size))

    snr_linear = None
    snr_db = None
    sinad_linear = None
    sinad_db = None

    if std_val > 0:
        if abs(mean_val) > 0:
            snr_linear = abs(mean_val) / std_val
            snr_db = 20.0 * math.log10(snr_linear)

        signal_rms = float(np.sqrt(np.mean(np.square(signal))))
        noise_rms = std_val
        if noise_rms > 0:
            sinad_linear = signal_rms / noise_rms
            sinad_db = 20.0 * math.log10(sinad_linear)

    return {
        "n_points": int(signal.size),
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "snr_linear": snr_linear,
        "snr_db": snr_db,
        "sinad_linear": sinad_linear,
        "sinad_db": sinad_db,
    }


def run_snr_sem_scan(folder_path: str, output_json_path=None):
    base = Path(folder_path)
    fs = 1000000000
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    results = []
    for fp in sorted(base.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".csv", ".json"}:
            continue
        if int(str(fp).split("_")[-1].split(".")[0]) in [25, 26, 27]:
            file_result = {"file": str(fp), "status": "ok", "error": None, "stats": None}
            try:
                signal = _get_signal_from_file(fp)
                if int(str(fp).split("_")[-1].split(".")[0]) in [4, 26]:
                    volt_fixed2 = signal
                elif int(str(fp).split("_")[-1].split(".")[0]) in [3, 7, 9]:
                    volt_fixed2 = patch_voltage_dips(signal, start_time=0.0, end_time=0.05, sample_rate=fs)
                elif int(str(fp).split("_")[-1].split(".")[0]) in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]:
                    volt_fixed2 = patch_voltage_peaks(signal, start_time=0.0, end_time=0.05, sample_rate=fs)
                stats = compute_stats(volt_fixed2)
                file_result["stats"] = stats
                print(f"{fp.name}: SNR linear={stats['snr_linear']}, SNR dB={stats['snr_db']}, SEM={stats['sem']}")
            except Exception as exc:
                file_result["status"] = "error"
                file_result["error"] = str(exc)
                print(f"{fp.name}: error: {file_result['error']}")

            results.append(file_result)

    if output_json_path is None:
        output_json_path = base / "snr_sem_results.json"

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"folder": str(base), "results": results}, f, indent=2)

    return results


def plot_snr_from_json(json_path: str):
    """Read snr_sem_results.json and plot SNR linear + SEM for C1 and C2 files separately, x=theoretical distance."""

    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Result JSON not found: {json_path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    results = data.get("results", [])

    treated_dir = Path(r"D:\uni\TPOP---Projet-1\data\jour 5\treatus_fleubuis\treated_data")
    treated_names = {f.stem for f in treated_dir.glob("SDS824X_HD_Binary_C[12]_*.*") if f.is_file()}
    values_map = {int(item["index"]): float(item["distance"]) for item in values}

    selected = []
    for entry in results:
        entry_file = Path(entry.get("file", "")).stem
        m = re.match(r"^SDS824X_HD_Binary_C([12])_(\d+)$", entry_file, re.IGNORECASE)
        if not m:
            continue

        kind = m.group(1)
        idx = int(m.group(2))

        # Accept entries by regex match (C1/C2), regardless of current treated_names membership.
        # If the treated_data folder contains disjoint set of indices, we still include C1/C2 by name.
        if entry_file not in treated_names:
            treated_names.add(entry_file)

        theoretical_distance = values_map.get(idx)
        if theoretical_distance is None:
            # support fallback to index numeric if mapping missing
            theoretical_distance = float(idx)

        stats = entry.get("stats") or {}
        snr_lin = stats.get("snr_linear")
        sem = stats.get("sem")
        if snr_lin is None or sem is None:
            continue

        selected.append({
            "kind": kind,
            "idx": idx,
            "distance": theoretical_distance,
            "file": entry_file,
            "snr_linear": snr_lin,
            "sem": sem,
        })

    if not selected:
        raise ValueError("No valid SNR entries found in JSON for treated_data files")

    # Combine C1 and C2 into one plot, each with separate series.
    grouped = {
        "1": [e for e in selected if e["kind"] == "1"],
        "2": [e for e in selected if e["kind"] == "2"],
    }

    plt.figure(figsize=(10, 5))
    plotted_any = False
    output = {}

    for kind, entries in grouped.items():
        if not entries:
            continue

        entries.sort(key=lambda e: e["distance"])
        distances = [e["distance"] for e in entries]
        snr_vals = [e["snr_linear"] for e in entries]
        sem_vals = [e["sem"] for e in entries]

        plt.errorbar(distances, snr_vals, yerr=sem_vals,
                     fmt='o-', capsize=3, label=f'SNR Linear C{kind}')

        output[f'C{kind}'] = {
            "distance": distances,
            "snr_linear": snr_vals,
            "sem": sem_vals,
            "files": [e["file"] for e in entries],
        }

        plotted_any = True

    if not plotted_any:
        raise ValueError("No valid C1/C2 data to plot")

    plt.title('SNR Linear with SEM for C1 and C2 (treated_data)')
    plt.xlabel('Theoretical distance')
    plt.ylabel('SNR Linear')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return output


if __name__ == "__main__":
    folder = r"D:\uni\TPOP---Projet-1\data\jour 5\treatus_fleubuis"
    out_file = r"D:\uni\TPOP---Projet-1\data\jour 5\treatus_fleubuis\treated_data\snr_sem_results.json"
    print(f"Scanning folder: {folder}")
    final = run_snr_sem_scan(folder, out_file)
    print(f"Saved JSON result to: {out_file}")
    print(json.dumps(final, indent=2))

    # try:
    #     plot_snr_from_json(out_file)
    # except Exception as e:
    #     print(f"Plotting error: {e}")
