import matplotlib.pyplot as plt
import sys
import os
 
 
def parse_spectrum_file(filepath):
    wavelengths = []
    intensities = []
    in_data_section = False
 
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
 
            if line == ">>>>>Begin Spectral Data<<<<<":
                in_data_section = True
                continue
 
            if in_data_section and line:
                # Data uses comma as decimal separator and tab as column separator
                parts = line.split("\t")
                if len(parts) == 2:
                    x = float(parts[0].replace(",", "."))
                    y = float(parts[1].replace(",", "."))
                    wavelengths.append(x)
                    intensities.append(y)
 
    return wavelengths, intensities
 
 
def plot_spectrum(filepath):
    wavelengths, intensities = parse_spectrum_file(filepath)
 
    if not wavelengths:
        print("No spectral data found in file.")
        return
 
    filename = os.path.basename(filepath)
 
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, intensities, linewidth=0.8, color="steelblue")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum — {filename}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    path = "D:\uni\TPOP-H26\projet2\data spectro\neon\neon_USB4F104161__0__11-46-31-085.txt"
    print(f"Usage: python plot_spectrum.py {path}")
 
    plot_spectrum(path)