"""Medan PM2,5 sintetis di wilayah kota berukuran 20 x 20 km (ilustrasi).

Konsentrasi latar ditambah beberapa 'titik panas' (kawasan industri, jalan
tol, simpul kemacetan) dan derau sensor murah. Satuan mikrogram per m^3.
"""
import numpy as np

PANAS = [(5, 14, 25, 2.5), (13, 6, 30, 3.0), (15, 15, 18, 4.0), (8, 4, 12, 2.0)]

def medan(x, y):
    z = 35 + 0.6 * x                                   # latar, naik ke timur
    for cx, cy, a, s in PANAS:
        z = z + a * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * s**2))
    return z

def sensor(n=40, sd=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 20, size=(n, 2))
    return X, medan(X[:, 0], X[:, 1]) + rng.normal(0, sd, n)

def jam(n_hari=3, seed=1):
    """Satu sensor: PM2,5 per jam dengan pola harian dan jam-jam hilang."""
    rng = np.random.default_rng(seed)
    t = np.arange(24 * n_hari, dtype=float)
    f = 45 + 15 * np.cos(2 * np.pi * (t - 7) / 24) + 8 * np.sin(2 * np.pi * t / 60)
    ada = ~((t >= 30) & (t < 44))                      # sensor mati 14 jam
    ada &= rng.random(t.size) < 0.6                    # dan data bolong acak
    return t[ada], f[ada] + rng.normal(0, 4, ada.sum()), t, f
