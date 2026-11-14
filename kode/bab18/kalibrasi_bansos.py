import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from data_bansos import buat_data, GARIS

X, log_c, w = buat_data()
y = (log_c < GARIS).astype(int)                          # 1 = miskin
X = StandardScaler().fit(X[:4000]).transform(X)
L, K, U = slice(0, 4000), slice(4000, 6000), slice(6000, 8000)   # latih, kalibrasi, uji #<1>

net = MLPClassifier((128, 128), alpha=0, max_iter=400, random_state=0).fit(X[L], y[L])
logit = lambda p: np.log(p / (1 - p))
z = logit(np.clip(net.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7))

def nll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def ece(p, y, B=10):                                     # expected calibration error
    b = np.minimum((p * B).astype(int), B - 1)
    return sum(abs(p[b == k].mean() - y[b == k].mean()) * np.mean(b == k)
               for k in range(B) if np.any(b == k))

T = minimize_scalar(lambda T: nll(1 / (1 + np.exp(-z[K] / T)), y[K]),
                    bounds=(0.05, 20), method="bounded").x          # temperatur  #<2>
platt = LogisticRegression(C=1e6).fit(z[K, None], y[K])             #<3>
iso = IsotonicRegression(out_of_bounds="clip").fit(z[K], y[K])      #<4>
P = {"tanpa kalibrasi": 1 / (1 + np.exp(-z[U])),
     f"temperatur T={T:.2f}": 1 / (1 + np.exp(-z[U] / T)),
     "Platt": platt.predict_proba(z[U, None])[:, 1],
     "isotonik": iso.predict(z[U])}
print(f"miskin di data uji: {y[U].mean():.3f}")
print("metode               akurasi    NLL    ECE  p>0,99 / p<0,01")
for nama, p in P.items():
    ekstrem = np.mean((p > 0.99) | (p < 0.01))
    print(f"{nama:20s} {np.mean((p > .5) == y[U]):7.3f} {nll(p, y[U]):6.3f}"
          f" {ece(p, y[U]):6.3f} {ekstrem:10.2f}")
np.savez("_kalibrasi.npz", y=y[U], *P.values())
