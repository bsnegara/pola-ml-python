"""Rumah tangga sintetis untuk proxy means test (PMT) bantuan sosial (ilustrasi).

Setiap rumah tangga punya ciri yang mudah diamati petugas (aset, kondisi
rumah, pendidikan kepala rumah tangga) dan pengeluaran per kapita yang tidak
diamati langsung. Wilayah 3T (terdepan, terluar, tertinggal) sengaja dibuat
lebih sedikit, lebih miskin, dan dengan hubungan aset--pengeluaran yang lebih
lemah. Angka TIDAK mencerminkan data DTKS/Regsosek yang sebenarnya.
"""
import numpy as np

WILAYAH = ["perkotaan", "perdesaan", "3T"]
GARIS = np.log(600)            # garis kemiskinan: Rp600 ribu/kapita/bulan

def buat_data(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.choice(3, n, p=[0.5, 0.4, 0.1])
    art = rng.integers(1, 9, n)                                   # anggota rumah tangga
    didik = np.clip(rng.normal(np.array([11, 8, 6])[w], 3), 0, 16)      # tahun sekolah KRT
    laten = rng.normal(0, 1, n)                                   # kesejahteraan tersembunyi
    lantai_tanah = (rng.normal(-laten - np.array([1.2, 0.6, -0.2])[w], 1) > 0).astype(float)
    listrik = (rng.normal(laten + np.array([2.5, 1.5, 0.0])[w], 1) > 0).astype(float)
    motor = (rng.normal(laten + np.array([0.6, 0.6, -0.4])[w], 1) > 0).astype(float)
    kulkas = (rng.normal(laten + np.array([0.3, -0.3, -1.0])[w], 1) > 0).astype(float)
    luas = np.exp(rng.normal(2.6 + 0.3 * laten - 0.08 * art, 0.3))  # m2 per kapita
    # pengeluaran: aset informatif di kota, kurang informatif di 3T
    kuat = np.array([0.35, 0.28, 0.15])[w]
    log_c = (np.array([7.05, 6.80, 6.60])[w] + kuat * laten + 0.025 * didik
             - 0.05 * art + rng.normal(0, np.array([0.18, 0.22, 0.40])[w]))
    X = np.c_[np.eye(3)[w], art, didik, lantai_tanah, listrik, motor, kulkas, np.log(luas)]
    return X, log_c, w
