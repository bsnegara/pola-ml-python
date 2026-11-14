import numpy as np

rng = np.random.default_rng(0)
f = lambda x: 10 * np.sin(2 * np.pi * x)        # pola musiman sebenarnya
pusat = np.linspace(0, 1, 24)
def Phi(x):                                     # 24 basis Gaussian + konstanta
    return np.column_stack([np.ones_like(x),
                            np.exp(-(x[:, None] - pusat)**2 / (2 * 0.1**2))])

x_uji = np.linspace(0, 1, 500)
L, N = 100, 25                                  # 100 "pasar", 25 pengamatan  #<1>
print(" ln(lambda)   bias^2   variansi   jumlah   galat uji")
for ln_lam in (-6, -3, -0.3, 1, 3):
    pred = np.empty((L, x_uji.size))
    galat = []
    for l in range(L):
        x = rng.uniform(0, 1, N)
        t = f(x) + rng.normal(0, 3, N)
        A = np.exp(ln_lam) * np.eye(25) + Phi(x).T @ Phi(x)
        w = np.linalg.solve(A, Phi(x).T @ t)        # ridge  #<2>
        pred[l] = Phi(x_uji) @ w
        galat.append(np.mean((pred[l] - f(x_uji) - rng.normal(0, 3, x_uji.size))**2))
    rerata = pred.mean(axis=0)                      # model "rata-rata"  #<3>
    bias2 = np.mean((rerata - f(x_uji))**2)
    var = np.mean(pred.var(axis=0))
    print(f"{ln_lam:10.1f} {bias2:8.2f} {var:10.2f} {bias2 + var:8.2f} {np.mean(galat):11.2f}")
