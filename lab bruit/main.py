import numpy as np
import matplotlib.pyplot as plt

def get_snr(signal):
    return np.mean(signal) / np.std(signal, ddof=1)

def get_sem(signal):
    return np.std(signal) / len(signal) ** 0.5

def tcl(signal, nb_echantillons, nb_moyenne):
    moyennes = []
    for i in range(nb_moyenne):
        moyennes.append(np.mean(np.random.choice(signal, nb_echantillons)))
    return np.asarray(moyennes)
    
def generate_val_distribution(values):
    counts, bin_edges = np.histogram(values, bins=30)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    plt.bar(
        bin_centers,
        counts,
        width=bin_width * 0.85,   # spacing between bars
        edgecolor='black',
        linewidth=1
    )
    stats_text = (
        f"snr = {get_snr(values)}\n"
        f"sem = {get_sem(values)}\n"
        f"mean = {np.mean(values)}\n"
    )
    plt.text(
        0.98, 0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
    )

    plt.xlabel("Plage de valeurs", fontsize=17)
    plt.ylabel("Nombre d'occurrences", fontsize=17)

    plt.tight_layout()
    plt.show()

def generate_mult_distribution(valuess):
    all_values = np.concatenate(valuess)       # flatten all data
    num_bins = 30
    bin_edges = np.linspace(np.min(all_values), np.max(all_values), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    titles = ["2 points", "5 points", "25 points", "100 points"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (ax, values) in enumerate(zip(axes, valuess)):
        # Histogram
        counts, _ = np.histogram(values, bins=bin_edges)

        ax.bar(
            bin_centers,
            counts,
            width=bin_width * 0.85,
            edgecolor='black',
            linewidth=1
        )
        # Labels
        ax.set_xlabel("Plage de valeurs", fontsize=14)
        ax.set_ylabel("Nombre d'occurrences", fontsize=14)
        ax.set_title(titles[i])

    #max_count = max(np.histogram(values, bins=bin_edges)[0].max() for values in valuess)
    #for ax in axes:
    #    ax.set_ylim(0, max_count * 1.1)

    plt.tight_layout()
    plt.show()

def generate_tcl_distribution(values):
    valuess = [tcl(values, 2, 2000), tcl(values, 5, 2000), tcl(values, 25, 2000), tcl(values, 100, 2000)]
    generate_mult_distribution(valuess)

def generate_mega_means_distribution(values):
    meaned_values = []
    total = 0
    n = 0
    first = True
    for value in values:
        if value[1] == 1:
            if first:
                first = False
            else:
                total += value[2]
                n += 1
        else:
            first = True
            if total != 0:
                meaned_values.append(total / n)
                total = 0
                n = 0
    generate_val_distribution(meaned_values)


if __name__ == '__main__':
    signal = np.genfromtxt("data TPOP 2/data_1.csv", delimiter=",", skip_header=1)
    #signal = np.loadtxt("data.csv", delimiter=",", skiprows=1)
    is_one = signal[:, 1] == 1

    # Find the first 1 in each contiguous series
    shifted = np.roll(is_one, 1)  # shift right
    shifted[0] = False            # first element has no previous
    first_of_series = is_one & (~shifted)  # first 1 in each series

    # Mask to take all ones except first in series
    mask = is_one & (~first_of_series)

    good_signal = signal[mask]
    time = good_signal[:, 0]
    voltage = good_signal[:, 2]

    #generate_val_distribution(voltage)
    #generate_tcl_distribution(voltage)
    generate_mega_means_distribution(signal)