import numpy as np
from data_sms import buat_korpus, kantong_kata

teks, t, _ = buat_korpus()
latih = np.arange(len(t)) < 1400
X, kata = kantong_kata(teks, latih)
Phi = np.column_stack([np.ones(len(t)), X])         # tambah konstanta
P, tl = Phi[latih], t[latih]
sigmoid = lambda a: 1 / (1 + np.exp(-a))

alpha = 1.0                                         # presisi prior Gaussian
w = np.zeros(P.shape[1])
for it in range(1, 30):                             # IRLS = metode Newton  #<1>
    y = sigmoid(P @ w)
    g = P.T @ (y - tl) + alpha * w                  # gradien  #<2>
    H = P.T @ (P * (y * (1 - y))[:, None]) + alpha * np.eye(len(w))  # Hessian
    langkah = np.linalg.solve(H, g)
    w -= langkah
    if np.abs(langkah).max() < 1e-8:
        break
print(f"IRLS konvergen dalam {it} iterasi")

p = sigmoid(Phi[~latih] @ w)
tu = t[~latih]
log_loss = np.mean(np.logaddexp(0, -(2 * tu - 1) * (Phi[~latih] @ w)))   #<3>
print(f"akurasi uji: {np.mean((p > 0.5) == tu):.3f} | log-loss uji: {log_loss:.3f}")
print(f"prediksi sangat yakin (p<0.001 atau p>0.999): {np.mean((p < 1e-3) | (p > 1 - 1e-3)):.0%}")
np.save("w_map.npy", w); np.save("H_map.npy", H)   # untuk aproksimasi Laplace
