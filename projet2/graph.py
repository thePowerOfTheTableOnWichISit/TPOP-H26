from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open("projet2/data/fo_pinhole_hg_2.bmp"))

profil = np.mean(img, axis=0)

plt.plot(profil)
plt.axvline(x=2521, color='y', linestyle='--')
plt.title("Profil d'intensité")
plt.show()
