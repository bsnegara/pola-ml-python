"""Profil sosial-ekonomi 38 provinsi sintetis (ilustrasi).

Setiap provinsi diberi kode wilayah (SUM, JAW, BNT, KAL, SUL, MAL, PAP) dan
nomor urut, bukan nama asli, karena angkanya dibangkitkan dari dua faktor laten
("pembangunan" dan "urbanisasi") ditambah derau, bukan data resmi.
"""
import numpy as np

INDIKATOR = ["miskin", "IPM", "RLS", "UHH", "pengangguran", "gini",
             "air_layak", "internet", "PDRB_log", "perkotaan", "stunting"]
WILAYAH = {"SUM": 10, "JAW": 6, "BNT": 3, "KAL": 5, "SUL": 6, "MAL": 2, "PAP": 6}

def buat_data(seed=0):
    rng = np.random.default_rng(seed)
    rerata = {"SUM": (0.2, 0.0), "JAW": (0.8, 1.2), "BNT": (-0.6, -0.3),
              "KAL": (0.4, 0.0), "SUL": (0.0, -0.3), "MAL": (-0.7, -0.4),
              "PAP": (-1.8, -0.9)}
    kode, F = [], []
    for w, n in WILAYAH.items():
        kode += [f"{w}-{i+1:02d}" for i in range(n)]
        F.append(rng.normal(rerata[w], 0.45, (n, 2)))
    F = np.vstack(F)                                   # faktor laten (38 x 2)
    # muatan: kolom 1 = pembangunan, kolom 2 = urbanisasi
    W = np.array([[-0.9, 0.0], [0.95, 0.2], [0.8, 0.4], [0.8, 0.0], [0.0, 0.8],
                  [0.1, 0.6], [0.8, 0.3], [0.6, 0.7], [0.5, 0.6], [0.3, 0.9],
                  [-0.8, -0.1]])
    psi = np.array([.3, .15, .3, .4, .5, .7, .4, .3, .6, .25, .45])  # derau unik
    Z = F @ W.T + rng.normal(0, 1, (38, 11)) * psi
    skala = np.array([[10, 4], [71, 4], [8.6, 1.1], [70, 2], [5, 1.5], [0.37, 0.03],
                      [88, 6], [70, 12], [3.8, 0.5], [50, 15], [22, 5]])
    X = skala[:, 0] + Z * skala[:, 1]
    return np.array(kode), X
