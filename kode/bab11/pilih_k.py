import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score as ari
from data_ipm import buat_data

X, z = buat_data()
A = StandardScaler().fit_transform(X)                      # keempat indikator
print(" K |    BIC   |    AIC   | silhouette K-means")
for K in range(1, 9):
    g = GaussianMixture(K, covariance_type="full", n_init=5, random_state=0).fit(A)  #<1>
    s = "  -" if K == 1 else f"{silhouette_score(A, KMeans(K, n_init=10, random_state=0).fit_predict(A)):.3f}"
    print(f"{K:2d} | {g.bic(A):8.1f} | {g.aic(A):8.1f} | {s}")

for jenis in ["full", "diag", "spherical"]:                 # bentuk kovarians  #<2>
    g = GaussianMixture(4, covariance_type=jenis, n_init=5, random_state=0).fit(A)
    print(f"GMM {jenis:9s}: BIC {g.bic(A):7.1f}, ARI {ari(z, g.predict(A)):.2f}")
km = KMeans(4, n_init=10, random_state=0).fit(A)
print(f"K-means      : ARI {ari(z, km.labels_):.2f}")
