import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

pixels = []
Intensite = []

# Fichier BMP

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open("data/FORANGE_IRIS_1.bmp"))

profil = np.mean(img, axis=0)

plt.plot(profil)
plt.title("Profil d'intensité")
plt.show()