import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_umkm import buat_data

X, y = buat_data()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
t = 2 * yl - 1                                     # label -1/+1

def stump_terbaik(X, t, w):                      # pohon satu tingkat (stump)  #<1>
    terbaik = (np.inf, None)
    for j in range(X.shape[1]):
        for ambang in np.unique(np.quantile(X[:, j], np.linspace(0.02, 0.98, 30))):
            for arah in (1, -1):
                pred = arah * np.where(X[:, j] > ambang, 1, -1)
                galat = w[pred != t].sum()          # galat tertimbang
                if galat < terbaik[0]:
                    terbaik = (galat, (j, ambang, arah))
    return terbaik

w = np.full(len(t), 1 / len(t))                    # bobot awal seragam
F_latih, F_uji = np.zeros(len(t)), np.zeros(len(yu))
for m in range(1, 201):
    eps, (j, amb, arah) = stump_terbaik(Xl, t, w)
    a = 0.5 * np.log((1 - eps) / eps)              # bobot pengklasifikasi  #<2>
    h = arah * np.where(Xl[:, j] > amb, 1, -1)
    w *= np.exp(-a * t * h); w /= w.sum()          # naikkan bobot yang salah  #<3>
    F_latih += a * h
    F_uji += a * arah * np.where(Xu[:, j] > amb, 1, -1)
    if m in (1, 5, 20, 50, 200):
        print(f"M = {m:3d} | galat tertimbang {eps:.3f} | AUC latih "
              f"{roc_auc_score(yl, F_latih):.3f} | uji {roc_auc_score(yu, F_uji):.3f}")
