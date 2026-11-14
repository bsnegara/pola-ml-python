import numpy as np

# Data hasil gabah yang sama dengan Bab 14 (ton/ha, sintetis).
x = np.array([5.8, 6.4, 5.1, 6.9, 5.5, 6.2, 7.1, 5.9, 6.6, 6.0])
N, xb = len(x), x.mean()
mu0, lam0, a0, b0 = 5.0, 1.0, 1.0, 1.0

rng = np.random.default_rng(0)
mu, tau, sampel = xb, 1.0, []
for t in range(20_000):                              #<1>
    # p(mu | tau, x): Gaussian
    lam = (lam0 + N) * tau
    mu = rng.normal((lam0 * mu0 + N * xb) / (lam0 + N), lam ** -0.5)
    # p(tau | mu, x): gamma
    a = a0 + (N + 1) / 2
    b = b0 + 0.5 * (np.sum((x - mu) ** 2) + lam0 * (mu - mu0) ** 2)
    tau = rng.gamma(a, 1 / b)
    sampel.append((mu, tau))
S = np.array(sampel[1000:])
print(f"Gibbs: E[mu] = {S[:, 0].mean():.4f}, sd[mu] = {S[:, 0].std():.4f},"
      f" E[tau] = {S[:, 1].mean():.4f}")
print("eksak: E[mu] = 6.0455, sd[mu] = 0.2462, E[tau] = 1.7998   (Bab 14)")
print("VB   : E[mu] = 6.0455, sd[mu] = 0.2247, E[tau] = 1.7998   (Bab 14)")
kecil = S[:, 1] < np.median(S[:, 1])                 # bentuk corong  #<2>
print(f"sd[mu] bila tau < median: {S[kecil, 0].std():.3f},"
      f" bila tau >= median: {S[~kecil, 0].std():.3f}")
