import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import os
from pathlib import Path

data_dir_env = os.getenv('DATADIR')
SCHEMATIC_DIR = Path(data_dir_env) / 'Schematics'
SCHEMATIC_DIR.mkdir(exist_ok=True)


def plot_gpr_samples(gpr_model, n_samples, ax):
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(x, single_prior, linestyle="--", alpha=0.7, )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(x, y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.1, color="black", )
    ax.set_xlabel("location")
    ax.set_ylabel("temperature")
    ax.set_ylim([-3, 3])


rng = np.random.RandomState(4)
number_of_observations = 15
X_train = rng.uniform(0, 5, number_of_observations).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5

kernel1 = 0.75 * RBF(length_scale=0.15, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=0.01)
kernel2 = 0.75 * Matern(length_scale=0.3, nu=0.5, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=0.01)
gpr1 = GaussianProcessRegressor(kernel=kernel1, optimizer=None, random_state=0)
gpr2 = GaussianProcessRegressor(kernel=kernel2, optimizer=None, random_state=0)

fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 12))

# plot prior
plot_gpr_samples(gpr1, n_samples=n_samples, ax=axs[0])
axs[0].set_title("(a) Samples from prior distribution", loc='left')

# plot posterior
gpr1.fit(X_train, y_train)
plot_gpr_samples(gpr1, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].set_title("(b) Samples from posterior distribution", loc='left')

# plot posterior
gpr2.fit(X_train, y_train)
plot_gpr_samples(gpr2, n_samples=n_samples, ax=axs[2])
axs[2].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[2].set_title("(c) Samples from posterior distribution alt. prior", loc='left')
plt.tight_layout()

plt.savefig(SCHEMATIC_DIR / 'gaussian_process.png')
plt.savefig(SCHEMATIC_DIR / 'gaussian_process.svg')
plt.savefig(SCHEMATIC_DIR / 'gaussian_process.pdf')

plt.close()
