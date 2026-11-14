import numpy as np
from data_provinsi import buat_data, INDIKATOR

kode, X = buat_data()
Z = (X - X.mean(0)) / X.std(0)                       # baku: satuan berbeda-beda  #<1>
S = Z.T @ Z / len(Z)                                 # = matriks korelasi
lam, U = np.linalg.eigh(S)                           # nilai & vektor eigen  #<2>
lam, U = lam[::-1], U[:, ::-1]                       # urutkan menurun
U *= np.sign(U[1])                                   # tanda: IPM bermuatan positif
porsi = lam / lam.sum()
print("porsi varians   :", " ".join(f"{p:.2f}" for p in porsi[:5]))
print("kumulatif       :", " ".join(f"{p:.2f}" for p in np.cumsum(porsi)[:5]))
print("muatan KU1, KU2 :")
for nama, u1, u2 in zip(INDIKATOR, U[:, 0], U[:, 1]):
    print(f"   {nama:13s} {u1:+.2f} {u2:+.2f}")
Y = Z @ U[:, :2]                                     # proyeksi ke 2 komponen  #<3>
G = Y @ U[:, :2].T                                   # rekonstruksi
print(f"galat rekonstruksi rata-rata: {np.mean((Z - G) ** 2):.3f}"
      f" (= jumlah eigen sisanya / D: {lam[2:].sum() / len(lam):.3f})")
_, s, Vt = np.linalg.svd(Z, full_matrices=False)     # jalur SVD  #<4>
print("eigen dari SVD sama:", np.allclose(s ** 2 / len(Z), lam))
for i in np.argsort(Y[:, 0])[[0, 1, -2, -1]]:
    print(f"   {kode[i]}: KU1 {Y[i, 0]:+.2f}, KU2 {Y[i, 1]:+.2f}")
