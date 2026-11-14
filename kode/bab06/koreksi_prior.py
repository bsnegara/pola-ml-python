import numpy as np
from sklearn.linear_model import LogisticRegression
from data_transaksi import buat_transaksi

X, y, hari = buat_transaksi()
latih, uji = hari < 120, hari >= 120
rng = np.random.default_rng(0)
pos = np.where(latih & (y == 1))[0]
neg = rng.choice(np.where(latih & (y == 0))[0], size=len(pos), replace=False)
seimbang = np.r_[pos, neg]                          # undersampling 50:50  #<1>
p = LogisticRegression(max_iter=1000).fit(X[seimbang], y[seimbang]).predict_proba(X[uji])[:, 1]

pi_latih, pi_baru = 0.5, y[latih].mean()            # prior saat latih vs nyata
r = p * pi_baru / pi_latih                          # koreksi prior (teorema Bayes)  #<2>
p_kor = r / (r + (1 - p) * (1 - pi_baru) / (1 - pi_latih))

def ece(p, t, k=10):                                # galat kalibrasi harapan  #<3>
    kelompok = np.minimum((p * k).astype(int), k - 1)
    return sum(np.abs(p[kelompok == b].mean() - t[kelompok == b].mean()) * np.mean(kelompok == b)
               for b in range(k) if np.any(kelompok == b))

print(f"prevalensi nyata di data uji: {y[uji].mean():.4f}")
for nama, q in [("tanpa koreksi", p), ("dengan koreksi", p_kor)]:
    ll = -np.mean(y[uji] * np.log(q) + (1 - y[uji]) * np.log(1 - q))
    print(f"{nama:14s}: rerata p {q.mean():.4f} | log-loss {ll:.4f} | "
          f"ECE {ece(q, y[uji]):.4f} | p>0.5: {np.sum(q > 0.5)}")
