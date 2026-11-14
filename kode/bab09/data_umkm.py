"""Data pinjaman UMKM sintetis (ilustrasi). Sekitar 10% gagal bayar.

Hubungan fitur dengan risiko sengaja dibuat nonlinear dan berinteraksi,
misalnya rasio cicilan berbahaya terutama untuk usaha muda, dan pencatatan
digital (QRIS) menurunkan risiko di sektor perdagangan.
"""
import numpy as np

FITUR = ["omzet_juta", "umur_usaha", "rasio_cicilan", "telat_12bln",
         "pakai_qris", "karyawan", "sektor", "di_kota"]
SEKTOR = ["kuliner", "dagang", "jasa", "kriya", "tani"]

def buat_data(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    omzet = np.exp(rng.normal(3.0, 0.8, n))                 # juta Rp per bulan
    umur = rng.gamma(2.0, 2.5, n)                           # tahun
    rasio = np.clip(rng.beta(2, 6, n), 0, 0.95)             # cicilan / laba
    telat = rng.poisson(0.4, n)
    qris = (rng.random(n) < 0.45).astype(int)
    karyawan = rng.poisson(1 + omzet / 15)
    sektor = rng.integers(0, 5, n)
    kota = (rng.random(n) < 0.6).astype(int)
    z = (-3.0
         + 3.0 * (rasio > 0.4) * (1 + 1.5 * (umur < 2))        # ambang rasio cicilan
         + 1.2 * np.minimum(telat, 3)                          # riwayat telat bayar
         - 0.8 * (omzet > 15)                                  # omzet besar lebih aman
         - 1.5 * qris * (sektor == 1)                          # QRIS membantu pedagang
         - 0.8 * (umur > 5)
         + 1.5 * (sektor == 4) * (1 - kota)                    # tani di desa: musiman
         + rng.normal(0, 0.3, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    X = np.column_stack([omzet, umur, rasio, telat, qris, karyawan, sektor, kota])
    return X, y
