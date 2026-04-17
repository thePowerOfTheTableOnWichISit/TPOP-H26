from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Etalonnage import etalonnage
from scipy.signal import find_peaks, savgol_filter
from matplotlib.ticker import MultipleLocator

def resolution_spectrometre(largeur_entree_m, focale_m, pas_reseau_m, angle_diffraction_deg, ordre=1, grandissement=1):
    beta = np.deg2rad(angle_diffraction_deg)
    dispersion_angulaire = ordre / (pas_reseau_m * np.cos(beta))
    return largeur_entree_m / (grandissement * focale_m * dispersion_angulaire)

filtres = [430, 450, 532, 533, 577]
conversion_px_nm = {}

for filtre in filtres:
    val, error = etalonnage(f"projet2/data/etalon_{filtre}nm.bmp", filtre)
    conversion_px_nm[filtre] = {"val": val, "error": error}

px_min = conversion_px_nm[450]["val"]
px_max = conversion_px_nm[577]["val"]
delta = (577 - 450) / (px_max - px_min)
offset = 450 - delta * px_min

Px = np.arange(3648)
Nm = delta * Px + offset + 2

img = np.array(Image.open("projet2/data/fo_pinhole_hg_2.bmp"))
profil = np.mean(img, axis=0)
profil_lisse = savgol_filter(profil, window_length=13, polyorder=3)

peaks, _ = find_peaks(profil, height=np.max(profil) * 0.2, distance=50)
if len(peaks) > 4:
    peaks = peaks[np.argsort(profil[peaks])[-4:]]

masque = np.ones(len(profil), dtype=bool)
for p in peaks:
    masque[max(0, p - 20):min(len(profil), p + 20)] = False

bruit = np.std(profil[masque])
print(f"Bruit (sigma) : {bruit:.2f}")

for p in peaks:
    print(f"Pic à {Nm[p]:.1f} nm : SNR = {profil[p] / bruit:.1f}")

plt.figure(figsize=(10, 6))
plt.plot(Nm, profil_lisse, color="black")

for p in peaks:
    x = Nm[p]
    y = profil[p]
    plt.plot(x, y, "o", markerfacecolor="none", markeredgecolor="gray", markersize=8)
    plt.text(x, y * 1.05, f"{x:.1f} nm", ha="center", fontsize=28)

plt.xlabel("Longueur d'onde (nm)", fontsize=28)
plt.ylabel("Intensité lumineuse", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(5))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spectre_hg.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()

res = (Nm.max() - Nm.min()) / 3648

print("\nPARAMÈTRES DU SPECTRO")
print(f"minimum : {Nm.min()}")
print(f"maximum : {Nm.max()}")
print(f"range du spectro : {Nm.max() - Nm.min()} nm")
print(f"résolution de la CCD : {res} nm par pixel")

res_spect = resolution_spectrometre(500e-6, 0.1, 1 / (1200e3), 5, 1, 1)
print(f"resolution du spectrometre : {res_spect * 1e9} nm")