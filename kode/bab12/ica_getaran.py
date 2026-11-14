import numpy as np
from sklearn.decomposition import FastICA, PCA

# Dua mesin di satu ruang (ilustrasi): press berdenyut, pompa bergetar gigi gergaji.
t = np.linspace(0, 1, 2000)
s1 = np.sign(np.sin(2 * np.pi * 7 * t))                       # denyut press
s2 = 2 * ((5 * t) % 1) - 1                                    # gergaji pompa
S = np.c_[s1, s2] + 0.05 * np.random.default_rng(0).normal(size=(2000, 2))
A = np.array([[1.0, 0.6], [0.5, 1.0]])                        # pencampuran  #<1>
X = S @ A.T                                                   # dua sensor

def korelasi_terbaik(Y):                                      # cocokkan sumber  #<2>
    C = np.abs(np.corrcoef(S.T, Y.T)[:2, 2:])
    return C.max(1)

Y_pca = PCA(2).fit_transform(X)
ica = FastICA(2, random_state=0, whiten="unit-variance")      #<3>
Y_ica = ica.fit_transform(X)
print("|korelasi| dengan sumber asli (sumber 1, sumber 2):")
print("  sensor mentah:", np.round(korelasi_terbaik(X), 3))
print("  PCA          :", np.round(korelasi_terbaik(Y_pca), 3))
print("  ICA          :", np.round(korelasi_terbaik(Y_ica), 3))
kurt = lambda y: np.mean(((y - y.mean()) / y.std()) ** 4) - 3  # non-Gaussian  #<4>
print("kurtosis lebih: sumber", np.round([kurt(s1), kurt(s2)], 2),
      "| campuran", np.round([kurt(X[:, 0]), kurt(X[:, 1])], 2))
