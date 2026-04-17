import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt

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

INDEX_TO_DISTANCE = {entry['index']: entry['distance'] for entry in values}

FILENAME_INDEX_RE = re.compile(r"(_(\d+))\.csv$", re.IGNORECASE)


def extract_index_from_filename(filename):
    # Match last integer before .csv, e.g. SDS824X_HD_Binary_C2_7.csv => 7
    base = os.path.basename(filename)
    m = FILENAME_INDEX_RE.search(base)
    if not m:
        raise ValueError(f"Could not extract index from filename '{filename}'")
    return int(m.group(2))


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'results' not in data:
        raise ValueError(f"JSON at {path} has no 'results' key")
    return data['results']


def prepare_series(results):
    c1 = []
    c2 = []

    for res in results:
        if res.get('status') != 'ok':
            continue
        file_path = res.get('file', '')
        if not file_path:
            continue

        group = None
        if '_C1_' in file_path or 'C1_' in file_path:
            group = 'C1'
        elif '_C2_' in file_path or 'C2_' in file_path:
            group = 'C2'
        else:
            continue

        try:
            idx = extract_index_from_filename(file_path)
        except ValueError:
            continue

        distance = INDEX_TO_DISTANCE.get(idx)
        if distance is None:
            # This may happen if some points are absent from values list
            print(f"warning: no theoretical distance for index={idx}, file={file_path}", file=sys.stderr)
            continue

        snr = res.get('stats', {}).get('snr_linear')
        if snr is None:
            continue

        if group == 'C1':
            c1.append((distance, snr))
        else:
            c2.append((distance, snr))

    c1.sort(key=lambda x: x[0])
    c2.sort(key=lambda x: x[0])
    return c1, c2


def plot_snr(c1, c2, out_path=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True)

    if not c1 and not c2:
        ax.text(0.5, 0.5, "No data for C1 or C2", ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if c1:
            distances_c1, snr_c1 = zip(*c1)
            ax.plot(distances_c1, snr_c1, linestyle='-', color='tab:blue', label='Capteur A')

        if c2:
            distances_c2, snr_c2 = zip(*c2)
            ax.plot(distances_c2, snr_c2, linestyle='--', color='tab:red', label='Capteur B')

        ax.set_xlabel('Distance entre capteur et laser (m)', fontsize=25)
        ax.set_ylabel('SNR', fontsize=25)
        ax.grid(False)
        ax.legend(fontsize=25)

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot snr_linear vs theoretical distance from snr_sem_results.json')
    parser.add_argument('--input', '-i', default=r'data/jour 5/treatus_fleubuis/treated_data/snr_sem_results.json',
                        help='Path to snr_sem_results.json')
    parser.add_argument('--output', '-o', default=None, help='Optional output image path')
    args = parser.parse_args()

    results = load_results(args.input)
    c1, c2 = prepare_series(results)

    # Focus requested file in description but plot all C1/C2 by index-distance mapping
    print(f"C1 points: {len(c1)}, C2 points: {len(c2)}")
    plot_snr(c1, c2, args.output)


if __name__ == '__main__':
    main()
