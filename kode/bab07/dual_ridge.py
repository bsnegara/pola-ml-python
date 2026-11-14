import numpy as np
import time

rng = np.random.default_rng(0)
N, M = 2000, 30                                    # banyak data, banyak fitur
Phi = rng.normal(size=(N, M)); t = Phi @ rng.normal(size=M) + rng.normal(0, 0.5, N)
lam = 1.0

t0 = time.perf_counter()                           # bentuk primal: M x M  #<1>
w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(M), Phi.T @ t)
t_primal = time.perf_counter() - t0

t0 = time.perf_counter()                           # bentuk dual: N x N  #<2>
K = Phi @ Phi.T                                    # matriks Gram k(x_n, x_m)
a = np.linalg.solve(K + lam * np.eye(N), t)
t_dual = time.perf_counter() - t0

phi_baru = rng.normal(size=(5, M))
print("prediksi primal:", np.round(phi_baru @ w, 4))
print("prediksi dual  :", np.round(phi_baru @ Phi.T @ a, 4))   # sum_n a_n k(x, x_n)  #<3>
print(f"waktu primal {t_primal*1e3:.1f} ms (matriks {M}x{M}), "
      f"dual {t_dual*1e3:.1f} ms (matriks {N}x{N})")
