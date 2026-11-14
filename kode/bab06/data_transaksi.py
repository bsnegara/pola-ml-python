"""Transaksi dompet digital sintetis (ilustrasi). Sekitar 1% tidak wajar.

Fitur: log nominal, jam (dalam bentuk sin/cos), jarak dari lokasi biasa (km,
log), merchant baru (0/1), banyak transaksi 24 jam terakhir, umur akun (log
hari). Pola penipuan bergeser pelan dari bulan ke bulan (drift).
"""
import numpy as np

NAMA_FITUR = ["log_nominal", "jam_sin", "jam_cos", "log_jarak",
              "merchant_baru", "frek_24j", "log_umur_akun"]

def buat_transaksi(n=60_000, laju=0.01, seed=0):
    rng = np.random.default_rng(seed)
    hari = np.sort(rng.integers(0, 180, n))            # enam bulan
    y = (rng.random(n) < laju).astype(int)
    geser = hari / 180                                 # drift: 0 -> 1
    jam = np.where(y == 1, rng.normal(2 + 3 * geser, 3, n), rng.normal(14, 4, n)) % 24
    X = np.column_stack([
        np.where(y == 1, rng.normal(13.0 - 1.0 * geser, 1.2, n), rng.normal(11.0, 1.3, n)),
        np.sin(2 * np.pi * jam / 24), np.cos(2 * np.pi * jam / 24),
        np.where(y == 1, rng.normal(3.0 - 1.2 * geser, 1.3, n), rng.normal(1.0, 1.0, n)),
        (rng.random(n) < np.where(y == 1, 0.6, 0.1)).astype(float),
        rng.poisson(np.where(y == 1, 4 - 2 * geser, 2)),
        np.where(y == 1, rng.normal(3.0, 1.5, n), rng.normal(5.5, 1.2, n)),
    ])
    return X, y, hari
