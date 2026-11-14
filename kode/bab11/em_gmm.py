import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from data_ipm import buat_data

X, z = buat_data()
X2 = X[:, [2, 3]]; X2 = (X2 - X2.mean(0)) / X2.std(0)

def em(X, mu, iter_maks=500, tol=1e-6):
    K, D = mu.shape
    pi, S = np.full(K, 1 / K), np.array([np.eye(D)] * K)
    riwayat = []
    for t in range(iter_maks):
        p = np.column_stack([pi[k] * mvn.pdf(X, mu[k], S[k]) for k in range(K)])
        riwayat.append(np.log(p.sum(1)).sum())             # log-likelihood  #<1>
        g = p / p.sum(1, keepdims=True)                    # langkah E  #<2>
        Nk = g.sum(0)                                      # langkah M  #<3>
        pi, mu = Nk / len(X), (g.T @ X) / Nk[:, None]
        for k in range(K):
            D_ = X - mu[k]
            S[k] = (g[:, k, None] * D_).T @ D_ / Nk[k] + 1e-6 * np.eye(D)
        if t > 0 and riwayat[-1] - riwayat[-2] < tol: break
    return pi, mu, S, g, riwayat

rng = np.random.default_rng(0)
for s in range(5):                                         # beberapa inisialisasi  #<4>
    *_, g, ll = em(X2, X2[rng.choice(len(X2), 4, replace=False)])
    print(f"inisialisasi acak {s}: {len(ll):3d} iterasi, log-lik. {ll[-1]:8.2f},"
          f" ARI {ari(z, g.argmax(1)):.2f}")
km = KMeans(4, n_init=10, random_state=0).fit(X2)
pi, mu, S, g, ll = em(X2, km.cluster_centers_)             # mulai dari K-means
print(f"mulai dari K-means : {len(ll):3d} iterasi, log-lik. {ll[-1]:8.2f},"
      f" ARI {ari(z, g.argmax(1)):.2f}")
print(f"K-means saja       : ARI {ari(z, km.labels_):.2f}")
print("log-lik. iterasi 0,1,2,5,10,20:", " ".join(f"{ll[t]:.0f}" for t in [0,1,2,5,10,20]))
ragu = np.sort(g, 1)[:, -1] < 0.6                          # responsibility terbesar < 0,6
print(f"kabupaten 'ragu' (responsibility maks. < 0,6): {ragu.sum()} dari {len(X)}")
