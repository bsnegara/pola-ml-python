import numpy as np
from data_ipm import buat_data

X, z = buat_data()
X2 = X[:, [2, 3]]                                    # RLS dan pengeluaran
X2 = (X2 - X2.mean(0)) / X2.std(0)                   # standardisasi  #<1>

def kmeans(X, K, rng, iter_maks=50):
    mu = X[rng.choice(len(X), K, replace=False)]     # inisialisasi acak
    riwayat = []
    for _ in range(iter_maks):
        d2 = ((X[:, None, :] - mu[None]) ** 2).sum(-1)
        r = d2.argmin(1)                             # langkah E: tugaskan  #<2>
        riwayat.append(d2[np.arange(len(X)), r].sum())
        mu_baru = np.array([X[r == k].mean(0) for k in range(K)])  # langkah M  #<3>
        if np.allclose(mu_baru, mu): break
        mu = mu_baru
    return mu, r, riwayat

rng = np.random.default_rng(3)
mu, r, J = kmeans(X2, 4, rng)
print("J per iterasi:", " ".join(f"{j:.0f}" for j in J))
hasil = [kmeans(X2, 4, np.random.default_rng(s))[2][-1] for s in range(20)]  #<4>
print(f"J akhir dari 20 inisialisasi: min {min(hasil):.1f}, maks {max(hasil):.1f}")
print("ukuran klaster:", np.bincount(r))
