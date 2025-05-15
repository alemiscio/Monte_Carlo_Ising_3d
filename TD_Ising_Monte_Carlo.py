import numpy as np

import time

from PIL import Image

import matplotlib.pyplot as plt

plt.ion()

def random_spin_field(N, M, L):
    return np.random.choice([-1, 1], size=(N, M, L))

def display_spin_field(field, z=0):
    slice_2d = field[:, :, z]
    return Image.fromarray(np.uint8((slice_2d + 1) * 0.5 * 255))

def display_spin_projection(field, axis=2):
    proj = np.mean(field, axis=axis)
    return Image.fromarray(np.uint8((proj + 1) * 0.5 * 255))

def _ising_update_3d(field, n, m, l, beta=0.4, h=0.0):
    N, M, L = field.shape
    total = (
        field[(n + 1) % N, m, l] +
        field[(n - 1) % N, m, l] +
        field[n, (m + 1) % M, l] +
        field[n, (m - 1) % M, l] +
        field[n, m, (l + 1) % L] +
        field[n, m, (l - 1) % L]
    )
    dE = 2 * field[n, m, l] * (total + h)
    if dE <= 0 or np.exp(-dE * beta) > np.random.rand():
        field[n, m, l] *= -1

def ising_step_3d(field, beta=0.4, h=0.0):
    N, M, L = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for l_offset in range(2):
                for n in range(n_offset, N, 2):
                    for m in range(m_offset, M, 2):
                        for l in range(l_offset, L, 2):
                            _ising_update_3d(field, n, m, l, beta, h)
    return field


def magnetization(field):
    return np.mean(field)

def two_point_correlation_r_3d(field, r, axis=2):
    shifted = np.roll(field, -r, axis=axis)
    return np.mean(field * shifted)

def two_point_correlation_r_connected_3d(field, r, axis=2):
    shifted = np.roll(field, -r, axis=axis)
    raw_corr = np.mean(field * shifted)
    mean_spin = np.mean(field)
    return raw_corr - mean_spin**2

def compute_two_point_r_profile_3d(field, r_max, axis=2):
    G = [two_point_correlation_r_connected_3d(field, r, axis=axis) for r in range(1, r_max+1)]
    return np.array(G)

def cft_prediction(r_values, A=1.0):
    Delta_sigma = 0.518148806  # from 2411.15300
    return A / (r_values ** (2* Delta_sigma))

def cft_prediction_epsilon(r_values, A=1.0):
    Delta_sigma = 1.41262528  # from 2411.15300
    return A / (r_values ** (2* Delta_sigma))


def plot_two_point_sigma(field, r_max, i, axis=2):
    tp = compute_two_point_r_profile_3d(field, r_max, axis=axis)
    A_fit = tp[0]
    G_cft = cft_prediction(np.arange(1, r_max + 1), A_fit)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-',
             label=fr'CFT')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $r$')
    plt.ylabel(r'$\langle \sigma(0) \sigma(r) \rangle$')
    plt.title(f'Comparison with CFT correlator after {i} steps')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"Ising3D_correlator_step_{i}.png")
    plt.close()



def export_two_point_data_txt(G,  filename='two_point_profile.txt'):
    r_values = np.arange(1, len(G) + 1)
    with open(filename, 'w') as f:
        for r, g in zip(r_values, G):
            f.write(f"{r:4d} {g:15.8f}\n")

def save_field_npy(field, filename="field_config.npy"):
    np.save(filename, field)

def load_field_npy(filename="field_config.npy"):
    return np.load(filename)

def compute_epsilon_field_3d(field):
    N, M, L = field.shape

    neighbors_sum = (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
    )
    epsilon = -field * neighbors_sum
    return epsilon

def energy_expectation_value_3d(field):
    return np.mean(compute_epsilon_field_3d(field)) / 2  # ← divide by 2 to avoid double counting


def two_point_correlation_epsilon_r_connected(field, r, axis=2):
    epsilon = compute_epsilon_field_3d(field)
    shifted = np.roll(epsilon, -r, axis=axis)
    raw_corr = np.mean(epsilon * shifted)
    mean_eps = np.mean(epsilon)
    return raw_corr - mean_eps**2

def compute_two_point_epsilon_profile(field, r_max, axis=2):
    return np.array([
        two_point_correlation_epsilon_r_connected(field, r, axis=axis)
        for r in range(1, r_max + 1)
    ])


def plot_two_point_epsilon(field, r_max, i, axis=2):
    tp = compute_two_point_epsilon_profile(field, r_max, axis=axis)
    A_fit = tp[0]
    G_cft = cft_prediction_epsilon(np.arange(1, r_max + 1), A_fit)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-', label=fr'CFT')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $r$')
    plt.ylabel(r'$\langle \epsilon(0) \epsilon(r) \rangle$')
    plt.title(f'Comparison with CFT correlator after {i} steps')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"Ising3D_correlator_e_step_{i}.png")
    plt.close()


