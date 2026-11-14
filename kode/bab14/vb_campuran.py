import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Empat kelompok titik 2D (sintetis, ilustrasi), 400 titik.
rng = np.random.default_rng(1)
pusat = np.array([[0, 0], [4, 1], [1, 4], [5, 5]])
X = np.vstack([rng.multivariate_normal(m, [[0.6, 0.2], [0.2, 0.5]], 100) for m in pusat])

K = 10                                                   # sengaja terlalu banyak
em = GaussianMixture(K, n_init=3, random_state=0).fit(X)  #<1>
vb = BayesianGaussianMixture(n_components=K, weight_concentration_prior=1e-3,
                             weight_concentration_prior_type="dirichlet_distribution",
                             n_init=3, max_iter=1000, random_state=0).fit(X)  #<2>
print("bobot EM :", " ".join(f"{w:.2f}" for w in np.sort(em.weights_)[::-1]))
print("bobot VB :", " ".join(f"{w:.2f}" for w in np.sort(vb.weights_)[::-1]))
print("komponen aktif (bobot > 1%): EM", np.sum(em.weights_ > 0.01),
      "| VB", np.sum(vb.weights_ > 0.01))
print(f"ELBO akhir VB (seluruh data): {vb.lower_bound_:.1f}")  #<3>
Xu = np.vstack([rng.multivariate_normal(m, [[0.6, 0.2], [0.2, 0.5]], 100) for m in pusat])
print(f"log-lik. uji per titik: EM {em.score(Xu):.3f}, VB {vb.score(Xu):.3f}")
