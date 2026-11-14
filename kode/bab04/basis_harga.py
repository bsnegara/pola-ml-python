import numpy as np
from data_harga import buat_data, fitur

t, y = buat_data()
latih, uji = t < 156, t >= 156                     # 3 tahun latih, 2025 uji  #<1>

def rmse(Phi):
    w, *_ = np.linalg.lstsq(Phi[latih], y[latih], rcond=None)  #<2>
    galat = Phi @ w - y
    return np.sqrt(np.mean(galat[latih]**2)), np.sqrt(np.mean(galat[uji]**2))

s = (t - 104) / 104                                # skala [-1, 1] agar stabil
model = {
    "polinom orde 9":         np.vander(s, 10, increasing=True),   #<3>
    "Fourier saja":           fitur(t, lebaran=False, tren=False),
    "Fourier + tren":         fitur(t, lebaran=False),
    "Fourier + tren + Lebaran": fitur(t),                          #<4>
}
print(f"{'model':26s} {'#param':>6s} {'RMSE latih':>11s} {'RMSE uji':>9s}")
for nama, Phi in model.items():
    r_l, r_u = rmse(Phi)
    print(f"{nama:26s} {Phi.shape[1]:6d} {r_l:11.2f} {r_u:9.2f}")
