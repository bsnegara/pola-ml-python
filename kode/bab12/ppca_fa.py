import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from data_provinsi import buat_data, INDIKATOR

kode, X = buat_data()
Z = (X - X.mean(0)) / X.std(0)
N, D = Z.shape
lam, U = np.linalg.eigh(Z.T @ Z / N); lam, U = lam[::-1], U[:, ::-1]

def ppca(M):                                         # solusi bentuk tertutup  #<1>
    s2 = lam[M:].mean()                              # rata-rata eigen yang dibuang
    W = U[:, :M] * np.sqrt(lam[:M] - s2)
    return W, s2

W, s2 = ppca(2)
C = W @ W.T + s2 * np.eye(D)                         # kovarians marginal  #<2>
ll = mvn.logpdf(Z, np.zeros(D), C).mean()
print(f"PPCA M=2: sigma^2 = {s2:.3f}, log-lik./provinsi = {ll:.3f}")
print(f"  PCA(2).score di scikit-learn: {PCA(2).fit(Z).score(Z):.3f}")
fa = FactorAnalysis(2, random_state=0).fit(Z)        # derau berbeda per variabel  #<3>
print("varians derau FA  :", " ".join(f"{v:.2f}" for v in fa.noise_variance_))
print(f"  terbesar: {INDIKATOR[fa.noise_variance_.argmax()]},"
      f" terkecil: {INDIKATOR[fa.noise_variance_.argmin()]}")
print(" M | log-lik. validasi silang PPCA |  FA")       #<4>
for M in range(1, 6):
    a = cross_val_score(PCA(M), Z, cv=5).mean()
    b = cross_val_score(FactorAnalysis(M, random_state=0), Z, cv=5).mean()
    print(f" {M} | {a:29.2f} | {b:5.2f}")
