import numpy as np

rng = np.random.default_rng(7)                    # data curah hujan Bab 1
f = lambda x: 200 + 150 * np.cos(2 * np.pi * x)
x = rng.uniform(0, 1, 12); t = f(x) + rng.normal(0, 40, 12)
x_uji = rng.uniform(0, 1, 1000); t_uji = f(x_uji) + rng.normal(0, 40, 1000)

def evidence(M, iterasi=500):
    Phi = np.vander(x, M + 1, increasing=True)
    N, D = Phi.shape
    lam0 = np.linalg.eigvalsh(Phi.T @ Phi)        # nilai eigen, dihitung sekali
    alpha, beta = 1e-3, 1e-3
    for _ in range(iterasi):                      # re-estimasi titik tetap  #<1>
        A = alpha * np.eye(D) + beta * Phi.T @ Phi
        m = beta * np.linalg.solve(A, Phi.T @ t)
        lam = beta * lam0
        gamma = np.sum(lam / (lam + alpha))       # banyak parameter efektif
        alpha = gamma / (m @ m)
        beta = (N - gamma) / np.sum((t - Phi @ m)**2)
    A = alpha * np.eye(D) + beta * Phi.T @ Phi
    m = beta * np.linalg.solve(A, Phi.T @ t)
    E = beta/2 * np.sum((t - Phi @ m)**2) + alpha/2 * m @ m
    ln_ev = (D/2*np.log(alpha) + N/2*np.log(beta) - E                 #<2>
             - np.linalg.slogdet(A)[1]/2 - N/2*np.log(2*np.pi))
    rmse = np.sqrt(np.mean((np.vander(x_uji, D, increasing=True) @ m - t_uji)**2))
    return ln_ev, gamma, 1/np.sqrt(beta), rmse

print(" M   ln evidence   gamma   sigma (mm)   RMSE uji (mm)")
for M in range(10):
    ln_ev, g, s, r = evidence(M)
    print(f"{M:2d} {ln_ev:13.2f} {g:7.2f} {s:12.1f} {r:15.1f}")
