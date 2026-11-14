"""Tekstur sintetis yang terinspirasi empat motif batik (ilustrasi, bukan citra asli).

parang  : garis-garis diagonal miring
kawung  : susunan lingkaran/elips berulang dalam kisi
truntum : titik-titik kecil tersebar acak
lereng  : pita diagonal lebar bergelombang
Setiap citra 32 x 32 piksel, dengan variasi skala, rotasi kecil, kontras, dan derau.
"""
import numpy as np

MOTIF = ["parang", "kawung", "truntum", "lereng"]

def citra(motif, rng, n=32, derau=1.4):
    y, x = np.mgrid[0:n, 0:n] / n
    sudut = rng.normal(0, 0.35)
    xr, yr = x*np.cos(sudut) - y*np.sin(sudut), x*np.sin(sudut) + y*np.cos(sudut)
    f = rng.uniform(4, 9)
    if motif == "parang":
        z = np.sin(2*np.pi*f*(xr + yr) / 1.4)
    elif motif == "kawung":
        u, v = (xr*f/1.6) % 1 - 0.5, (yr*f/1.6) % 1 - 0.5
        z = np.where(u**2/0.12 + v**2/0.06 < 1, 1.0, -1.0) + np.where(v**2/0.12 + u**2/0.06 < 1, 1.0, 0.0)
    elif motif == "truntum":
        z = -np.ones((n, n))
        for cx, cy in rng.uniform(0, n, size=(int(f * 4), 2)).astype(int):
            z[max(cy-1, 0):cy+2, max(cx-1, 0):cx+2] = 1.0
    else:
        z = np.sign(np.sin(2*np.pi*(f/3)*(xr + yr) + 1.5*np.sin(2*np.pi*3*(xr - yr))))
    z = rng.uniform(0.6, 1.4) * z + rng.normal(0, derau, (n, n))
    return z

def fitur(z):
    """Energi spektrum Fourier dalam 4 cincin frekuensi x 4 orientasi = 16 fitur."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(z - z.mean())))**2
    n = z.shape[0]; ky, kx = np.mgrid[-n//2:n//2, -n//2:n//2]
    r, th = np.hypot(kx, ky), np.mod(np.arctan2(ky, kx), np.pi)
    hasil = []
    for r0, r1 in [(1, 3), (3, 6), (6, 10), (10, 16)]:
        for k in range(4):
            m = (r >= r0) & (r < r1) & (th >= k*np.pi/4) & (th < (k+1)*np.pi/4)
            hasil.append(np.log(F[m].sum() + 1e-9))
    return np.array(hasil)

def buat_data(n_per_motif=150, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, m in enumerate(MOTIF):
        for _ in range(n_per_motif):
            X.append(fitur(citra(m, rng))); y.append(k)
    return np.array(X), np.array(y)
