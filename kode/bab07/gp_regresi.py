import numpy as np
from data_pm25 import jam

x, t, x_semua, f_semua = jam()                     # jam pengamatan, PM2,5

def kernel(a, b, theta=(20.0, 4.0)):               # RBF: amplitudo^2, skala  #<1>
    amp2, ell = theta
    return amp2**2 * np.exp(-(a[:, None] - b[None, :])**2 / (2 * ell**2))

def gp_prediksi(x, t, x_baru, theta, sd_derau):
    m = t.mean()                                   # rerata konstanta
    C = kernel(x, x, theta) + sd_derau**2 * np.eye(len(x))     #<2>
    L = np.linalg.cholesky(C)                      # stabil dan cepat
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, t - m))
    k = kernel(x, x_baru, theta)
    rerata = m + k.T @ alpha                       # m(x*) = k^T C^-1 t  #<3>
    v = np.linalg.solve(L, k)
    varians = theta[0]**2 - np.sum(v**2, axis=0)     # c - k^T C^-1 k
    return rerata, np.sqrt(varians)                  # sd fungsi f (tanpa derau)

print(f"pengamatan: {len(x)} dari {len(x_semua)} jam")
r, s = gp_prediksi(x, t, x_semua, (20.0, 4.0), 4.0)
for jam_ in (20, 36, 40, 60):                      # 36 dan 40: sensor mati
    print(f"jam {jam_:2d}: prediksi {r[jam_]:5.1f} +- {2*s[jam_]:4.1f} | sebenarnya {f_semua[jam_]:5.1f}")
cakup = np.mean(np.abs(f_semua - r) < 2 * s)
print(f"fungsi sebenarnya di dalam pita +-2sd: {cakup:.0%} dari semua jam")
