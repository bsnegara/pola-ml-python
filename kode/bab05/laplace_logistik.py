import numpy as np
from data_sms import buat_korpus, kantong_kata

teks, t, _ = buat_korpus()
latih = np.arange(len(t)) < 1400
baru = ["Kak ini foto paketnya, tolong cek dulu",
        "Selamat ulang tahun kak, semoga sehat",
        "Lowongan kerja paruh waktu, gaji harian 300rb, minat?",   # jenis baru  #<1>
        "Bu, transfer dulu ya ke rek BRI, penting"]
X, _ = kantong_kata(list(teks) + baru, np.r_[latih, np.zeros(len(baru), bool)])
Phi = np.column_stack([np.ones(len(X)), X])[-len(baru):]

w = np.load("w_map.npy")                            # mode posterior (IRLS)
S = np.linalg.inv(np.load("H_map.npy"))             # kovarians Laplace = H^-1  #<2>
sigmoid = lambda a: 1 / (1 + np.exp(-a))

mu_a = Phi @ w                                      # rerata aktivasi
var_a = np.sum(Phi @ S * Phi, axis=1)               # varians aktivasi
kappa = 1 / np.sqrt(1 + np.pi * var_a / 8)          # aproksimasi probit  #<3>
rng = np.random.default_rng(0)                      # pembanding Monte Carlo
W = rng.multivariate_normal(w, S, size=20_000)
p_mc = sigmoid(Phi @ W.T).mean(axis=1)

print("pesan  kata dikenal   p_MAP   p_Bayes   (Monte Carlo)   sd aktivasi")
for i in range(len(baru)):
    print(f"{i+1:5d} {int(X[-len(baru)+i].sum()):13d} {sigmoid(mu_a[i]):7.3f} "
          f"{sigmoid(kappa[i]*mu_a[i]):9.3f} {p_mc[i]:12.3f} {np.sqrt(var_a[i]):13.2f}")
