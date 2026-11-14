import numpy as np
from data_sms import buat_korpus, kantong_kata

teks, t, _ = buat_korpus()
latih = np.arange(len(t)) < 1400                    # 1400 latih, 600 uji
X, kata = kantong_kata(teks, latih)
print(f"kosakata: {len(kata)} kata")

# Kuadrat terkecil dengan target 0/1  #<1>
Phi = np.column_stack([np.ones(len(t)), X])
b = np.linalg.lstsq(Phi[latih], t[latih], rcond=None)[0]
y = Phi[~latih] @ b
print(f"kuadrat terkecil: keluaran uji di [{y.min():.2f}, {y.max():.2f}], "
      f"{np.mean((y < 0) | (y > 1)):.0%} di luar [0, 1]")

# Proyeksi ke satu dimensi: selisih rerata vs Fisher  #<2>
X0, X1 = X[latih & (t == 0)], X[latih & (t == 1)]
m0, m1 = X0.mean(0), X1.mean(0)
Sw = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)
arah = {"selisih rerata": m1 - m0,
        "Fisher": np.linalg.solve(Sw + 1e-1 * np.eye(len(kata)), m1 - m0)}  #<3>
for nama, w in arah.items():
    z0, z1 = X0 @ w, X1 @ w
    J = (z1.mean() - z0.mean())**2 / (z0.var() + z1.var())   # kriteria Fisher
    zu = X[~latih] @ w
    ambang = 0.5 * (z0.mean() + z1.mean())
    akurasi = np.mean((zu > ambang) == t[~latih])
    print(f"{nama:15s}: J = {J:6.2f} | akurasi uji = {akurasi:.3f}")
