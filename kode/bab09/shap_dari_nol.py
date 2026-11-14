import numpy as np
from itertools import combinations
from math import factorial
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from data_umkm import buat_data, FITUR, SEKTOR

X, y = buat_data()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
gb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=65, max_depth=3,
                                    categorical_features=[6], random_state=0).fit(Xl, yl)
f = lambda A: gb.decision_function(A)              # keluaran dalam log-odds

imp = permutation_importance(gb, Xu, yu, scoring="roc_auc", n_repeats=10, random_state=0)  #<1>
urut = np.argsort(-imp.importances_mean)
print("kepentingan permutasi (turunnya AUC uji):")
print("  " + ", ".join(f"{FITUR[j]} {imp.importances_mean[j]:.3f}" for j in urut[:4]))

x = Xu[np.argmax(f(Xu))]                           # pemohon paling berisiko di data uji
latar = Xl[np.random.default_rng(0).choice(len(Xl), 100, replace=False)]
d = len(FITUR)
def v(S):                                          # nilai koalisi: fitur S dari x  #<2>
    Z = latar.copy(); Z[:, list(S)] = x[list(S)]
    return f(Z).mean()
nilai = {S: v(S) for k in range(d + 1) for S in combinations(range(d), k)}
phi = np.zeros(d)
for j in range(d):                                 # rumus Shapley eksak  #<3>
    for S, vS in nilai.items():
        if j not in S:
            bobot = factorial(len(S)) * factorial(d - len(S) - 1) / factorial(d)
            phi[j] += bobot * (nilai[tuple(sorted(S + (j,)))] - vS)
print(f"pemohon: rasio {x[2]:.2f}, umur {x[1]:.1f} th, telat {int(x[3])}x, omzet {x[0]:.1f} jt, "
      f"{SEKTOR[int(x[6])]}, {'kota' if x[7] else 'desa'}")
print(f"p(gagal) = {gb.predict_proba(x[None])[0, 1]:.2f}; dasar (rata-rata latar) {nilai[()]:+.2f} log-odds")
for j in np.argsort(-np.abs(phi))[:5]:
    print(f"  {FITUR[j]:14s} {phi[j]:+.2f}")
print(f"dasar + jumlah SHAP = {nilai[()] + phi.sum():+.2f} = f(x) = {f(x[None])[0]:+.2f}")  #<4>
