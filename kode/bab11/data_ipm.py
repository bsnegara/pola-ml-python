"""Indikator pembangunan manusia 514 kabupaten/kota sintetis (ilustrasi).

Empat kelompok tersembunyi dengan sebaran berbeda: perkotaan, perdesaan maju,
perdesaan berkembang, dan daerah tertinggal. Angka tidak mewakili daerah nyata.
"""
import numpy as np

FITUR = ["UHH", "HLS", "RLS", "pengeluaran"]   # tahun, tahun, tahun, juta Rp/thn
KELOMPOK = ["perkotaan", "perdesaan maju", "perdesaan berkembang", "tertinggal"]

def buat_data(seed=0):
    rng = np.random.default_rng(seed)
    n = [98, 160, 196, 60]
    mu = np.array([[73.5, 14.8, 11.0, 16.0],
                   [71.8, 13.3, 8.8, 11.4],
                   [69.6, 12.5, 7.3, 8.9],
                   [65.0, 10.6, 5.2, 6.4]])
    sd = np.array([[0.9, 0.5, 0.6, 2.6],
                   [0.8, 0.4, 0.5, 0.9],
                   [1.0, 0.4, 0.5, 0.7],
                   [2.0, 1.2, 1.1, 0.8]])
    rho = [0.5, 0.85, 0.85, 0.8]           # korelasi antarindikator dalam kelompok
    X, z = [], []
    for k in range(4):
        R = np.full((4, 4), rho[k]); np.fill_diagonal(R, 1)
        S = R * np.outer(sd[k], sd[k])
        X.append(rng.multivariate_normal(mu[k], S, n[k])); z += [k] * n[k]
    X, z = np.vstack(X), np.array(z)
    p = rng.permutation(len(z))
    return X[p], z[p]
