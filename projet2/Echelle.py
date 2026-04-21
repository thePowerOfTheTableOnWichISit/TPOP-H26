from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from matplotlib.ticker import MultipleLocator
from Etalonnage import etalonnage


def charger_profil(chemin_image):
    img = np.array(Image.open(chemin_image))
    return np.mean(img, axis=0)

def calculer_fwhm(profil, x, y_peak):
    demi_hauteur = y_peak / 2

    # gauche
    i_gauche = x
    while i_gauche > 0 and profil[i_gauche] > demi_hauteur:
        i_gauche -= 1

    # droite
    i_droite = x
    while i_droite < len(profil)-1 and profil[i_droite] > demi_hauteur:
        i_droite += 1

    return i_droite - i_gauche

def calculer_calibration(conversion_px_nm, lambda_1=450, lambda_2=577, sigma_lambda=5):
    x1 = conversion_px_nm[lambda_1]["val"]
    x2 = conversion_px_nm[lambda_2]["val"]

    a = (lambda_2 - lambda_1) / (x2 - x1)
    b = lambda_1 - a * x1

    sigma_a = np.sqrt(sigma_lambda**2 + sigma_lambda**2) / abs(x2 - x1)
    sigma_b = np.sqrt(sigma_lambda**2 + (x1 * sigma_a)**2)

    return a, b, sigma_a, sigma_b, x1, x2


def detecter_pics(profil, nb_max=5):
    peaks, _ = find_peaks(profil, height=np.max(profil) * 0.2, distance=50)
    if len(peaks) > nb_max:
        peaks = peaks[np.argsort(profil[peaks])[-nb_max:]]
    return np.sort(peaks)


def estimer_bruit(profil, peaks, demi_largeur=20):
    masque = np.ones(len(profil), dtype=bool)
    for p in peaks:
        gauche = max(0, p - demi_largeur)
        droite = min(len(profil), p + demi_largeur)
        masque[gauche:droite] = False
    return np.std(profil[masque])


def associer_raies(lambda_mesurees, raies_reference):
    indices_ref_utilises = set()
    associations = []

    for lm in lambda_mesurees:
        diffs = np.abs(raies_reference - lm)
        ordre = np.argsort(diffs)

        for idx in ordre:
            if idx not in indices_ref_utilises:
                indices_ref_utilises.add(idx)
                associations.append(raies_reference[idx])
                break

    return np.array(associations)


filtres = [430, 450, 532, 533, 577]
sigma_lambda_calibration = 5
raies_reference_hg = np.array([404.7, 435.8, 546.1, 577.0, 579.0])

conversion_px_nm = {}
for filtre in filtres:
    val, error = etalonnage(f"projet2/data/etalon_{filtre}nm.bmp", filtre)
    conversion_px_nm[filtre] = {"val": val, "error": error}

a, b, sigma_a, sigma_b, px_450, px_577 = calculer_calibration(
    conversion_px_nm,
    lambda_1=450,
    lambda_2=577,
    sigma_lambda=sigma_lambda_calibration
)

pixels = np.arange(3648)
decalage_nm = 2
longueurs_onde = a * pixels + b + decalage_nm

pixel_min = pixels[0]
pixel_max = pixels[-1]

lambda_min = longueurs_onde[pixel_min]
lambda_max = longueurs_onde[pixel_max]
plage_spectrale = lambda_max - lambda_min

sigma_min = sigma_b
sigma_max = np.sqrt((pixel_max * sigma_a)**2 + sigma_b**2)
sigma_plage = abs(pixel_max - pixel_min) * sigma_a

resolution_ccd = a
sigma_resolution_ccd = sigma_a

profil = charger_profil("projet2/data/fo_pinhole_hg_2.bmp")
profil_lisse = savgol_filter(profil, window_length=13, polyorder=3)

peaks = detecter_pics(profil, nb_max=5)
bruit = estimer_bruit(profil, peaks)

lambda_pics = longueurs_onde[peaks]
raies_associees = associer_raies(lambda_pics, raies_reference_hg)

resultats = []
for p, lambda_mesuree, lambda_ref in zip(peaks, lambda_pics, raies_associees):
    sigma_lambda_i = np.sqrt((p * sigma_a)**2 + sigma_b**2)
    erreur_relative = (lambda_mesuree - lambda_ref) / lambda_ref
    sigma_erreur_relative = sigma_lambda_i / lambda_ref
    snr = profil[p] / bruit

    resultats.append({
        "pixel": p,
        "lambda_mesuree": lambda_mesuree,
        "sigma_lambda": sigma_lambda_i,
        "lambda_ref": lambda_ref,
        "erreur_relative": erreur_relative,
        "sigma_erreur_relative": sigma_erreur_relative,
        "snr": snr,
        "intensite": profil[p]
    })

fwhm_pixels = []

for p in peaks:
    largeur_px = calculer_fwhm(profil, p, profil[p])
    fwhm_pixels.append(largeur_px)

fwhm_pixels = np.array(fwhm_pixels)

# conversion en nm
fwhm_nm = fwhm_pixels * a

R_exp = np.mean(fwhm_nm)
sigma_R = np.std(fwhm_nm, ddof=1) / np.sqrt(len(fwhm_nm))


print(f"Bruit (sigma) : {bruit:.2f}")
print("\nRAIES DU MERCURE IDENTIFIÉES\n")

for i, r in enumerate(resultats, start=1):
    print(f"Raie {i}")
    print(f"Pixel : {r['pixel']}")
    print(f"Mesurée : {r['lambda_mesuree']:.1f} ± {r['sigma_lambda']:.1f} nm")
    print(f"Référence : {r['lambda_ref']:.1f} nm")
    print(f"Écart relatif : {r['erreur_relative'] * 100:.2f} ± {r['sigma_erreur_relative'] * 100:.2f} %")
    print(f"SNR : {r['snr']:.1f}")
    print("-" * 45)

plt.figure(figsize=(10, 6))
plt.plot(longueurs_onde, profil_lisse, color="black")

for r in resultats:
    x = r["lambda_mesuree"]
    y = r["intensite"]
    s = r["sigma_lambda"]

    plt.plot(x, y, "o", markerfacecolor="none", markeredgecolor="gray", markersize=8)
    plt.text(x, y * 1.05, f"{x:.1f} ± {s:.1f} nm", ha="center", fontsize=18)

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

print("\nPARAMÈTRES DU SPECTRO")
print(f"Minimum : {lambda_min:.2f} ± {sigma_min:.2f} nm")
print(f"Maximum : {lambda_max:.2f} ± {sigma_max:.2f} nm")
print(f"Plage spectrale : {plage_spectrale:.2f} ± {sigma_plage:.2f} nm")
print(f"Résolution de la CCD : {resolution_ccd:.4f} ± {sigma_resolution_ccd:.4f} nm/pixel")
print(f"Pente a : {a:.6f} ± {sigma_a:.6f} nm/pixel")
print(f"Offset b : {b:.2f} ± {sigma_b:.2f} nm")
print("\nRÉSOLUTION EXPÉRIMENTALE")
print(f"FWHM moyennes : {R_exp:.2f} ± {sigma_R:.2f} nm")
