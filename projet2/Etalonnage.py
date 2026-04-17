import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


def lire_image_gris(chemin):
    img = Image.open(chemin)
    img = np.array(img)
    img_g = img.astype(float)
    return img_g


def extraire_profil_spectral(img_g):
    profil = np.sum(img_g, axis=0)
    return profil


def centroid(x, y):
    somme = np.sum(y)
    if somme <= 0:
        return np.mean(x)
    return np.sum(x * y) / somme


def estimer_bruit(profil, indices_pics, demi_fenetre=20):
    masque = np.ones(len(profil), dtype=bool)
    for p in indices_pics:
        gauche = max(0, p - demi_fenetre)
        droite = min(len(profil), p + demi_fenetre + 1)
        masque[gauche:droite] = False

    zone_bruit = profil[masque]

    if len(zone_bruit) < 5:
        return np.std(profil)

    bruit = np.std(zone_bruit)
    if bruit <= 0:
        bruit = 1e-12

    return bruit


def calculer_positions_centroides_et_incertitudes(profil, indices_pics, demi_fenetre=20, bruit=None):
    positions = []
    incertitudes = []
    fwhm_list = []
    snr_list = []
    n_eff_list = []

    for p in indices_pics:
        gauche = max(0, p - demi_fenetre)
        droite = min(len(profil), p + demi_fenetre + 1)

        x = np.arange(gauche, droite, dtype=float)
        y = profil[gauche:droite].copy()

        fond_local = np.min(y)
        y = y - fond_local
        y[y < 0] = 0

        x_centre = centroid(x, y)

        if np.sum(y) <= 0:
            positions.append(np.mean(x))
            incertitudes.append(np.nan)
            fwhm_list.append(np.nan)
            snr_list.append(np.nan)
            n_eff_list.append(0.0)
            continue

        moments2 = np.sum(y * (x - x_centre) ** 2) / np.sum(y)
        sigma_pic = np.sqrt(max(moments2, 0.0))
        fwhm = 2.354820045 * sigma_pic

        amplitude = np.max(y)
        n_eff = np.sum(y)

        if bruit is None or bruit <= 0:
            snr = np.nan
        else:
            snr = amplitude / bruit

        if n_eff > 0:
            incertitude = sigma_pic / np.sqrt(n_eff)
        else:
            incertitude = np.nan

        positions.append(x_centre)
        incertitudes.append(incertitude)
        fwhm_list.append(fwhm)
        snr_list.append(snr)
        n_eff_list.append(n_eff)

    return {
        "positions": np.array(positions),
        "incertitudes": np.array(incertitudes),
        "fwhm": np.array(fwhm_list),
        "snr": np.array(snr_list),
        "n_eff": np.array(n_eff_list),
    }


def calibrer_lineaire(pixels, longueurs_onde):
    coeffs = np.polyfit(pixels, longueurs_onde, 1)
    a, b = coeffs
    return a, b


def etalonnage(nom_fichier, longueur_onde_connue):

    sigma_lissage = 2.0
    hauteur_min_relative = 0.2
    distance_min_pic = 20
    demi_fenetre_centroid = 20

    img_g = lire_image_gris(nom_fichier)

    profil_brut = extraire_profil_spectral(img_g)
    profil_corrige = profil_brut - np.min(profil_brut)
    profil_lisse = gaussian_filter1d(profil_corrige, sigma=sigma_lissage)

    hauteur_min = hauteur_min_relative * np.max(profil_lisse)
    indices_pics, proprietes = find_peaks(
        profil_lisse,
        height=hauteur_min,
        distance=distance_min_pic
    )

    bruit = estimer_bruit(
        profil_lisse,
        indices_pics,
        demi_fenetre=demi_fenetre_centroid
    )

    resultats_pics = calculer_positions_centroides_et_incertitudes(
        profil_lisse,
        indices_pics,
        demi_fenetre=demi_fenetre_centroid,
        bruit=bruit
    )

    positions_centroides = resultats_pics["positions"]
    incertitudes_centroides = resultats_pics["incertitudes"]
    fwhm_pics = resultats_pics["fwhm"]
    snr_pics = resultats_pics["snr"]
    n_eff_pics = resultats_pics["n_eff"]

    i_principal = np.argmax(profil_lisse[indices_pics])
    pic_principal = indices_pics[i_principal]
    position_centroide = positions_centroides[i_principal]
    incertitude_centroide = incertitudes_centroides[i_principal]
    fwhm_principal = fwhm_pics[i_principal]
    snr_principal = snr_pics[i_principal]
    n_eff_principal = n_eff_pics[i_principal]

    print("################")
    print(f"Lambda de réf : {longueur_onde_connue}")
    print(f"Pixel du pic (entier) : {pic_principal}")
    print(f"Pixel du pic (centroïde) : {position_centroide:.3f}")
    print(f"Incertitude sur le centroïde : {incertitude_centroide:.4f} pixel")
    print(f"FWHM estimée : {fwhm_principal:.4f} pixel")
    print(f"SNR estimé : {snr_principal:.2f}")
    print(f"N effectif : {n_eff_principal:.2f}")

    return position_centroide, incertitude_centroide
