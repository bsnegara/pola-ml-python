import numpy as np
from sklearn.svm import SVR

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-10, 10, 100)); t = np.sinc(x / np.pi) + rng.normal(0, 0.1, 100)
xu = np.linspace(-10, 10, 500); fu = np.sinc(xu / np.pi)       # sin(x)/x
K = lambda a, b: np.exp(-(a[:, None] - b[None, :])**2 / (2 * 1.5**2))

svr = SVR(kernel="rbf", gamma=1 / (2 * 1.5**2), C=10, epsilon=0.1).fit(x[:, None], t)  #<1>
r_svr = np.sqrt(np.mean((svr.predict(xu[:, None]) - fu)**2))

# Relevance vector machine: satu presisi prior alpha_i per basis  #<2>
Phi = np.c_[np.ones(100), K(x, x)]
alpha, beta = np.ones(Phi.shape[1]), 100.0
for _ in range(300):
    aktif = alpha < 1e6                                        # basis yang belum dipangkas
    P = Phi[:, aktif]
    S = np.linalg.inv(np.diag(alpha[aktif]) + beta * P.T @ P)
    m = beta * S @ P.T @ t
    gamma = 1 - alpha[aktif] * np.diag(S)
    alpha[aktif] = gamma / m**2                                # re-estimasi seperti Bab 4  #<3>
    beta = (100 - gamma.sum()) / np.sum((t - P @ m)**2)
aktif = alpha < 1e6
r_rvm = np.sqrt(np.mean((np.c_[np.ones(500), K(xu, x)][:, aktif] @ m - fu)**2))
print(f"SVR: {len(svr.support_):3d} support vector   | RMSE {r_svr:.3f}")
print(f"RVM: {aktif.sum() - aktif[0]:3d} vektor relevansi  | RMSE {r_rvm:.3f} | "
      f"sd derau {1/np.sqrt(beta):.3f} (sebenarnya 0.100)")
