import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from data_bansos import buat_data, GARIS, WILAYAH

X, y, w = buat_data()                                    # y = log pengeluaran per kapita
L, K, U = slice(0, 4000), slice(4000, 6000), slice(6000, 8000)
model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      random_state=0).fit(X[L], y[L])
skor = np.abs(y[K] - model.predict(X[K]))                # skor ketidaksesuaian  #<1>
yu, wu, mu = y[U], w[U], model.predict(X[U])

def kuantil(s, alfa=0.1):                                #<2>
    n = len(s)
    return np.quantile(s, np.ceil((n + 1) * (1 - alfa)) / n, method="higher")

q = kuantil(skor)
q_w = np.array([kuantil(skor[w[K] == k]) for k in range(3)])    # Mondrian  #<3>
print(f"target cakupan 90%; lebar selang satu kuantil: x{np.exp(2 * q):.2f}")
print("               cakupan dengan kuantil ...\n"
      "wilayah       n   gabungan  per-wilayah   lebar")
for k in range(3):
    m = wu == k
    c1 = np.mean(np.abs(yu[m] - mu[m]) <= q)
    c2 = np.mean(np.abs(yu[m] - mu[m]) <= q_w[k])
    print(f"{WILAYAH[k]:10s} {m.sum():4d} {c1:10.3f} {c2:12.3f}   x{np.exp(2 * q_w[k]):.2f}")

lo, hi = mu - q_w[wu], mu + q_w[wu]                       # keputusan tiga arah  #<4>
putus = np.where(hi < GARIS, "layak", np.where(lo > GARIS, "tidak", "verif"))
miskin = yu < GARIS
print("\nwilayah    exclusion  inclusion | exclusion  inclusion  verifikasi")
print("           (titik, ambang garis) | (konformal; verifikasi dikecualikan)")
for k in range(3):
    m = wu == k
    ex1 = np.mean(mu[m & miskin] > GARIS)                 # miskin tetapi ditolak
    in1 = np.mean(mu[m & ~miskin] < GARIS)                # tidak miskin tetapi diterima
    ex2 = np.mean(putus[m & miskin] == "tidak")
    in2 = np.mean(putus[m & ~miskin] == "layak")
    print(f"{WILAYAH[k]:10s} {ex1:8.3f} {in1:10.3f} | {ex2:8.3f} {in2:10.3f}"
          f" {np.mean(putus[m] == 'verif'):10.3f}")
np.savetxt("_konformal.txt", np.c_[yu, mu, lo, hi, wu][:400], fmt="%.4f")
