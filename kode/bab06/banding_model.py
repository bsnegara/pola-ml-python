import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from data_transaksi import buat_transaksi

X, y, hari = buat_transaksi()
latih, uji = hari < 120, hari >= 120
pA = LogisticRegression(max_iter=1000).fit(X[latih], y[latih]).predict_proba(X[uji])[:, 1]
pB = HistGradientBoostingClassifier(random_state=0).fit(X[latih], y[latih]).predict_proba(X[uji])[:, 1]
t = y[uji]

# Uji McNemar: hanya pasangan yang berbeda pendapat yang informatif  #<1>
bA, bB = (pA > 0.5) == t, (pB > 0.5) == t
n01, n10 = np.sum(bA & ~bB), np.sum(~bA & bB)
print(f"McNemar: A benar B salah = {n01}, A salah B benar = {n10}, "
      f"p = {binomtest(n01, n01 + n10).pvalue:.3f}")

# Bootstrap selisih AP: ambil ulang transaksi uji dengan pengembalian  #<2>
rng = np.random.default_rng(0)
selisih = []
for _ in range(2000):
    i = rng.integers(0, len(t), len(t))
    if t[i].sum() == 0:
        continue
    selisih.append(average_precision_score(t[i], pA[i]) - average_precision_score(t[i], pB[i]))
lo, hi = np.percentile(selisih, [2.5, 97.5])                          #<3>
print(f"AP logistik - AP boosting = {average_precision_score(t, pA) - average_precision_score(t, pB):.3f}, "
      f"selang bootstrap 95% [{lo:.3f}, {hi:.3f}]")
