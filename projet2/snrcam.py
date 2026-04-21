"""
snr_analysis.py
---------------
Compute the Signal-to-Noise Ratio (SNR) for BMP files containing
optical / spectrometer signal data (e.g. IRIS scans).
 
File format assumptions
-----------------------
- Grayscale BMP (mode "L"), shape (rows, cols).
- Multiple rows are independent acquisitions (or scan lines) of the
  same 1-D spectral / intensity signal.
- A bright peak (or peaks) rides on top of a noise floor / background.
 
SNR methods available
---------------------
1. peak_to_noise  : SNR = peak_signal / RMS_noise
                    Signal  = max value of the averaged trace.
                    Noise   = RMS of the background (pixels far from
                              the peak, defined by a threshold).
 
2. signal_to_bg   : SNR = (peak - background) / std_background
                    Classic spectroscopy definition.  Background mean
                    and std are estimated from the low-signal region.
 
3. row_variance   : SNR estimated from repeated rows.
                    Signal  = mean of the column-wise mean across rows.
                    Noise   = mean of the column-wise std across rows
                              (pixel-level shot/read noise per column).
 
Usage
-----
    python snr_analysis.py <file.bmp> [file2.bmp ...]
 
    python snr_analysis.py FORANGE_IRIS_1.bmp --method all --plot
"""
 
import argparse
import sys
from pathlib import Path
 
import numpy as np
from PIL import Image
 
 
# ─────────────────────────── helpers ────────────────────────────────────────
 
def load_bmp(path: str) -> np.ndarray:
    """Open a BMP and return a float64 numpy array (rows × cols)."""
    img = Image.open(path)
    if img.mode not in ("L", "I", "F"):
        # Convert RGB/RGBA to grayscale if needed
        img = img.convert("L")
    return np.array(img, dtype=np.float64)
 
 
def averaged_signal(arr: np.ndarray) -> np.ndarray:
    """Return the column-wise mean across all rows → 1-D signal."""
    return arr.mean(axis=0)
 
 
def background_mask(signal: np.ndarray, peak_fraction: float = 0.2) -> np.ndarray:
    """
    Boolean mask of 'background' pixels: those whose value is below
    peak_fraction × max(signal).  These are used to estimate noise.
    """
    threshold = signal.max() * peak_fraction
    return signal < threshold
 
 
# ─────────────────────────── SNR methods ────────────────────────────────────
 
def snr_peak_to_noise(arr: np.ndarray) -> dict:
    """
    SNR = peak_value / RMS_of_noise_floor
 
    Noise floor = RMS of background pixels in the averaged signal.
    """
    sig = averaged_signal(arr)
    bg_mask = background_mask(sig)
    if bg_mask.sum() < 10:
        raise ValueError("Too few background pixels — adjust peak_fraction.")
 
    peak      = sig.max()
    noise_rms = np.sqrt(np.mean(sig[bg_mask] ** 2))
    snr       = peak / noise_rms if noise_rms > 0 else np.inf
 
    return {
        "method"        : "peak_to_noise",
        "peak_value"    : peak,
        "peak_col"      : int(sig.argmax()),
        "noise_rms"     : noise_rms,
        "snr_linear"    : snr,
        "snr_dB"        : 20 * np.log10(snr),
        "bg_pixels_used": int(bg_mask.sum()),
    }
 
 
def snr_signal_to_bg(arr: np.ndarray) -> dict:
    """
    SNR = (peak - mean_background) / std_background
 
    Classic spectroscopy / analytical-chemistry definition.
    """
    sig     = averaged_signal(arr)
    bg_mask = background_mask(sig)
    if bg_mask.sum() < 10:
        raise ValueError("Too few background pixels — adjust peak_fraction.")
 
    peak      = sig.max()
    bg_mean   = sig[bg_mask].mean()
    bg_std    = sig[bg_mask].std(ddof=1)
    net_signal = peak - bg_mean
    snr        = net_signal / bg_std if bg_std > 0 else np.inf
 
    return {
        "method"        : "signal_to_bg",
        "peak_value"    : peak,
        "peak_col"      : int(sig.argmax()),
        "bg_mean"       : bg_mean,
        "bg_std"        : bg_std,
        "net_signal"    : net_signal,
        "snr_linear"    : snr,
        "snr_dB"        : 20 * np.log10(snr) if snr > 0 else -np.inf,
        "bg_pixels_used": int(bg_mask.sum()),
    }
 
 
def snr_row_variance(arr: np.ndarray) -> dict:
    """
    SNR from repeated rows:
        signal  = column-wise mean  (the 'true' signal level per pixel)
        noise   = column-wise std   (fluctuation = shot + read noise)
 
    Returns the median SNR across all columns, and separately for the
    peak region only.
    """
    if arr.shape[0] < 2:
        raise ValueError("Need ≥ 2 rows for row-variance SNR estimation.")
 
    col_mean = arr.mean(axis=0)
    col_std  = arr.std(axis=0, ddof=1)
 
    # Avoid division by zero
    valid = col_std > 0
    snr_per_col = np.where(valid, col_mean / col_std, np.nan)
 
    # Peak region: columns above 50% of max mean
    peak_mask = col_mean >= col_mean.max() * 0.5
 
    return {
        "method"            : "row_variance",
        "snr_median_all"    : float(np.nanmedian(snr_per_col)),
        "snr_median_peak"   : float(np.nanmedian(snr_per_col[peak_mask])),
        "snr_mean_peak"     : float(np.nanmean(snr_per_col[peak_mask])),
        "snr_dB_median_peak": 20 * np.log10(float(np.nanmedian(snr_per_col[peak_mask]))),
        "peak_cols"         : int(peak_mask.sum()),
        "rows_used"         : arr.shape[0],
    }
 
 
# ─────────────────────────── reporting ──────────────────────────────────────
 
def print_result(result: dict, filepath: str):
    width = 50
    print("=" * width)
    print(f"  File   : {Path(filepath).name}")
    print(f"  Method : {result['method']}")
    print("-" * width)
    for k, v in result.items():
        if k in ("method",):
            continue
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")
    print("=" * width)
    print()
 
 
def maybe_plot(arr: np.ndarray, filepath: str, results: list):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping plot.")
        return
 
    sig     = averaged_signal(arr)
    bg_mask = background_mask(sig)
    cols    = np.arange(len(sig))
 
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(f"SNR Analysis — {Path(filepath).name}", fontsize=13)
 
    # ── top: signal profile ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(cols, sig, lw=0.8, color="steelblue", label="Averaged signal")
    ax.fill_between(cols, sig, where=bg_mask, alpha=0.3,
                    color="gray", label="Background region")
    peak_col = int(sig.argmax())
    ax.axvline(peak_col, color="red", ls="--", lw=1,
               label=f"Peak @ col {peak_col}  ({sig.max():.1f})")
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Intensity (DN)")
    ax.set_title("Column-averaged intensity profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
 
    # ── bottom: per-row variance image ───────────────────────────────
    ax = axes[1]
    ax.imshow(arr, aspect="auto", cmap="gray", interpolation="nearest")
    ax.axvline(peak_col, color="red", ls="--", lw=1)
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row")
    ax.set_title("Raw BMP data (all rows)")
 
    plt.tight_layout()
 
    out = Path(filepath).with_suffix(".snr_plot.png")
    plt.savefig(out, dpi=150)
    print(f"[plot] Saved → {out}")
    plt.show()
 
 
# ─────────────────────────── CLI ────────────────────────────────────────────
 
def analyse(filepath: str, methods: list, do_plot: bool):
    arr = load_bmp(filepath)
    rows, cols = arr.shape
    print(f"\nLoaded: {filepath}  [{rows} rows × {cols} cols, "
          f"dtype={arr.dtype}, range={arr.min():.0f}–{arr.max():.0f}]")
 
    all_results = []
    dispatch = {
        "peak_to_noise" : snr_peak_to_noise,
        "signal_to_bg"  : snr_signal_to_bg,
        "row_variance"  : snr_row_variance,
    }
 
    chosen = list(dispatch.keys()) if "all" in methods else methods
    for name in chosen:
        try:
            result = dispatch[name](arr)
            all_results.append(result)
            print_result(result, filepath)
        except Exception as exc:
            print(f"  [{name}] Error: {exc}\n")
 
    if do_plot:
        maybe_plot(arr, filepath, all_results)
 
    return all_results
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Compute SNR for BMP optical / spectrometer signal files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="One or more .bmp files")
    parser.add_argument(
        "--method", "-m",
        nargs="+",
        default=["all"],
        choices=["peak_to_noise", "signal_to_bg", "row_variance", "all"],
        help="SNR method(s) to use (default: all)",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Show (and save) signal profile plots (requires matplotlib)",
    )
    args = parser.parse_args()
 
    path = "D:\uni\TPOP-H26\projet2\data\fo_iris_NEON.bmp"
    analyse(path, args.method, args.plot)
 
 
if __name__ == "__main__":
    main()
 