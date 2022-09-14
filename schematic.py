import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

data_dir_env = os.getenv('DATADIR')
SCHEMATIC_DIR = Path(data_dir_env) / 'Schematics'
SCHEMATIC_DIR.mkdir(exist_ok=True)


def plot_with_uncertainty(ax, years, data, uncertainty, tag):
    colors = []
    for i in range(len(years)):
        color = '#000000'
        if years[i] > 2019:
            color = '#DD0000'
        colors.append(color)
        ax.plot([years[i], years[i]], [data[i] - uncertainty[i], data[i] + uncertainty[i]], color=color)

    ax.scatter(years, data, color=colors)
    ax.set_ylim(0.25, 1.1)
    ax.set_xlim(2010, 2023)
    ax.text(2010.5,1.0,tag)


np.random.seed(571)

years = [y for y in range(1990, 2023)]
years = np.array(years)
data = 0.02 * (years - 1990)
noise = np.random.normal(0.0, 0.1, len(years))

uncertainty1 = np.zeros(len(years)) + 0.01
uncertainty2 = np.zeros(len(years)) + 0.034
uncertainty3 = np.zeros(len(years)) + 0.1
uncertainty4 = np.zeros(len(years)) + 0.5

data = data + noise

data[-1] = 0.8634225013944912
data[-2] = 0.9114547504086538
data[-3] = 0.8193039278231542

data[-5] = data[-5] - 0.1

fig, axs = plt.subplots(2,2)

plot_with_uncertainty(axs[0,0], years, data, uncertainty1,'(a)')
plot_with_uncertainty(axs[0,1], years, data, uncertainty3, '(b)')
plot_with_uncertainty(axs[1,0], years, data, uncertainty2, '(c)')
plot_with_uncertainty(axs[1,1], years, data, uncertainty4, '(d)')

plt.savefig(SCHEMATIC_DIR / 'uncertainty.png')
plt.savefig(SCHEMATIC_DIR / 'uncertainty.pdf')
plt.close()
