import numpy as np
from data_harga import buat_data, fitur, pola_harga

t, y = buat_data()
latih = t < 156
Phi, tl = fitur(t[latih]), y[latih]
N, D = Phi.shape

alpha, beta = 1.0, 1.0
for _ in range(100):                              # alpha, beta dari evidence  #<1>
    S = np.linalg.inv(alpha * np.eye(D) + beta * Phi.T @ Phi)
    m = beta * S @ Phi.T @ tl
    gamma = D - alpha * np.trace(S)
    alpha, beta = gamma / (m @ m), (N - gamma) / np.sum((tl - Phi @ m)**2)
print(f"alpha = {alpha:.4f}, sigma derau = {1/np.sqrt(beta):.2f} ribu Rp, "
      f"gamma = {gamma:.2f} dari {D} parameter")

def prediksi(t_baru):                             # distribusi prediktif  #<2>
    phi = fitur(np.atleast_1d(t_baru))
    return phi @ m, np.sqrt(1/beta + np.sum(phi @ S * phi, axis=1))

uji = ~latih
r, s = prediksi(t[uji])
cakup = np.mean(np.abs(y[uji] - r) < 1.96 * s)    # cakupan selang 95%  #<3>
print(f"tahun 2025: {cakup:.1%} pengamatan jatuh di selang prediksi 95%")

print("minggu           prediksi   +-2 sd   sebenarnya")
for nama, tb in [("Lebaran 2025", 169), ("Juli 2025", 182),
                 ("Juli 2027", 286), ("Juli 2032", 546)]:            #<4>
    r, s = prediksi(tb)
    print(f"{nama:14s} {r[0]:10.1f} {2*s[0]:8.1f} {pola_harga(tb):12.1f}")
