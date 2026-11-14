import numpy as np

rng = np.random.default_rng(7)
f = lambda x: 200 + 150 * np.cos(2 * np.pi * x)
x_latih = rng.uniform(0, 1, 12); t_latih = f(x_latih) + rng.normal(0, 40, 12)
x_uji = rng.uniform(0, 1, 1000); t_uji = f(x_uji) + rng.normal(0, 40, 1000)

M = 9
Phi = lambda x: np.vander(x, M + 1, increasing=True)  # 1, x, ..., x^M
beta = 1 / 40**2                                      # presisi derau  #<1>

print(" alpha      lambda=alpha/beta   RMSE uji (mm)   |w| maks")
for alpha in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:          # presisi prior  #<2>
    A = alpha * np.eye(M + 1) + beta * Phi(x_latih).T @ Phi(x_latih)
    w_map = np.linalg.solve(A, beta * Phi(x_latih).T @ t_latih)  #<3>
    rmse = np.sqrt(np.mean((Phi(x_uji) @ w_map - t_uji) ** 2))
    print(f"{alpha:6.0e}   {alpha / beta:17.1e}   {rmse:13.1f}   {np.abs(w_map).max():9.0f}")
