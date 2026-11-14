import numpy as np
from data_sms import buat_korpus, kantong_kata

teks, t, _ = buat_korpus()
latih = np.arange(len(t)) < 1400
X, kata = kantong_kata(teks, latih)
Xl, tl = X[latih], t[latih]

# Bernoulli naive Bayes: p(kata muncul | kelas), dengan prior Beta(1, 1)  #<1>
pi = np.array([np.mean(tl == 0), np.mean(tl == 1)])
mu = np.array([(Xl[tl == k].sum(0) + 1) / ((tl == k).sum() + 2) for k in (0, 1)])

def log_posterior_odds(X):                          #<2>
    lp = [np.log(pi[k]) + X @ np.log(mu[k]) + (1 - X) @ np.log(1 - mu[k]) for k in (0, 1)]
    return lp[1] - lp[0]                            # a = ln p(C1|x) - ln p(C0|x)

a = log_posterior_odds(X[~latih])
p = 1 / (1 + np.exp(-a))                            # sigmoid logistik  #<3>
log_loss = np.mean(np.logaddexp(0, -(2 * t[~latih] - 1) * a))   # stabil numerik
print(f"akurasi uji: {np.mean((p > 0.5) == t[~latih]):.3f} | log-loss uji: {log_loss:.3f}")
print(f"prediksi sangat yakin (p<0.001 atau p>0.999): {np.mean((p < 1e-3) | (p > 1 - 1e-3)):.0%}")

bobot = np.log(mu[1] / mu[0])                       # sumbangan tiap kata  #<4>
urut = np.argsort(bobot)
print("paling 'penipuan':", ", ".join(kata[urut[-8:][::-1]]))
print("paling 'normal'  :", ", ".join(kata[urut[:8]]))
