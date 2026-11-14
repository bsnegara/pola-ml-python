import numpy as np

# Hasil gabah kering (ton/ha) dari 10 petak uji (sintetis, ilustrasi).
x = np.array([5.8, 6.4, 5.1, 6.9, 5.5, 6.2, 7.1, 5.9, 6.6, 6.0])
N, xb = len(x), x.mean()
mu0, lam0, a0, b0 = 5.0, 1.0, 1.0, 1.0         # prior Normal-Gamma  #<1>

E_tau = 1.0                                     # tebakan awal E[tau]
for it in range(5):                             # pembaruan bergantian  #<2>
    muN = (lam0 * mu0 + N * xb) / (lam0 + N)    # q(mu) = N(muN, 1/lamN)
    lamN = (lam0 + N) * E_tau
    E_mu, E_mu2 = muN, muN ** 2 + 1 / lamN
    aN = a0 + (N + 1) / 2                       # q(tau) = Gam(aN, bN)
    bN = b0 + 0.5 * (np.sum(x ** 2) - 2 * E_mu * x.sum() + N * E_mu2
                     + lam0 * (E_mu2 - 2 * mu0 * E_mu + mu0 ** 2))
    E_tau = aN / bN
    print(f"iterasi {it}: E[mu] = {muN:.4f}, sd[mu] = {lamN ** -0.5:.4f},"
          f" E[tau] = {E_tau:.4f}")

# posterior eksak Normal-Gamma (Bab 3) untuk pembanding  #<3>
lamE, muE = lam0 + N, (lam0 * mu0 + N * xb) / (lam0 + N)
aE = a0 + N / 2
bE = b0 + 0.5 * (np.sum((x - xb) ** 2) + lam0 * N * (xb - mu0) ** 2 / (lam0 + N))
sd_mu_eksak = np.sqrt(bE / (aE - 1) / lamE)     # marginal mu: Student-t
print(f"eksak     : E[mu] = {muE:.4f}, sd[mu] = {sd_mu_eksak:.4f}, E[tau] = {aE / bE:.4f}")
