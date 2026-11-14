"""Citra potongan daun padi sintetis 32 x 32 RGB untuk 4 kelas (ilustrasi).

0 sehat, 1 blas (bercak belah ketupat berpusat abu-abu), 2 bercak cokelat
(bercak bulat kecil), 3 hawar daun bakteri (menguning dari tepi). Pola dibuat
sangat disederhanakan dan bukan pengganti citra lapangan.
"""
import numpy as np

KELAS = ["sehat", "blas", "bercak cokelat", "hawar"]

def citra(k, rng, S=32):
    yy, xx = np.mgrid[0:S, 0:S] / S
    hijau = np.array([0.25, 0.55, 0.2]) + rng.normal(0, 0.04, 3)
    I = np.ones((S, S, 3)) * hijau
    I *= (1 + 0.08 * np.sin(2 * np.pi * (xx * rng.uniform(5, 8) + rng.uniform())))[..., None]  # urat daun
    if k == 1:                                        # blas: belah ketupat
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.uniform(0.1, 0.9, 2); a, b = rng.uniform(0.04, 0.08), rng.uniform(0.1, 0.2)
            d = np.abs(xx - cx) / a + np.abs(yy - cy) / b
            I[d < 1] = [0.45, 0.3, 0.15]; I[d < 0.5] = [0.7, 0.7, 0.65]
    if k == 2:                                        # bercak cokelat: bulat kecil
        for _ in range(rng.integers(2, 6)):
            cx, cy = rng.uniform(0.05, 0.95, 2); r = rng.uniform(0.025, 0.05)
            I[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = [0.45, 0.25, 0.1]
    if k == 3:                                        # hawar: menguning dari satu tepi
        sisi = rng.integers(2); lebar = rng.uniform(0.08, 0.3)
        tepi = (xx if sisi == 0 else 1 - xx) + 0.05 * np.sin(yy * rng.uniform(6, 12))
        I[tepi < lebar] = [0.8, 0.75, 0.35]
    for _ in range(rng.integers(0, 4)):               # kotoran/bayangan di semua kelas
        cx, cy = rng.uniform(0, 1, 2); r = rng.uniform(0.02, 0.05)
        I[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] *= 0.5
    I *= rng.uniform(0.7, 1.3)                        # pencahayaan
    I += rng.normal(0, 0.06, I.shape)                 # derau kamera
    return np.clip(I, 0, 1).astype(np.float32)

def buat_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    X = np.stack([citra(k, rng) for k in y]).transpose(0, 3, 1, 2)   # N, C, H, W
    return X, y
