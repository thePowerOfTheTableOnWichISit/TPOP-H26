import numpy as np

def get_snr(signal):
    return np.mean(signal) / np.std(signal, ddof=1)

def get_sem(signal):
    return np.std(signal) / len(signal) ** 0.5

def tcl(signal, nb_echantillons, nb_moyenne):
    moyennes = []
    for i in range(nb_moyenne):
        moyennes.append(np.mean(np.random.choice(signal, nb_echantillons)))
    return np.asarray(moyennes)
    


if __name__ == '__main__':
    signal = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
    #signal = np.loadtxt("data.csv", delimiter=",", skiprows=1)
    time = signal[:, 0]
    voltage = signal[:, 1]

    #np.where(signal > 1, signal, 0)
    signal_cut = signal[signal > 1]
