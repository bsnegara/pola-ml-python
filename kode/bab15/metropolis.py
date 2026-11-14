import numpy as np
from scipy.special import expit

# 12 kabupaten (sintetis): akses sanitasi layak (dibakukan) dan kasus stunting.
rng = np.random.default_rng(3)
x = rng.normal(0, 1, 12); n = rng.integers(60, 200, 12)
y = rng.binomial(n, expit(-1.3 - 0.4 * x))

def log_post(th):                                   # logit p = a + b x, prior N(0, 2^2)
    a, b = th; eta = a + b * x
    return np.sum(y * eta - n * np.logaddexp(0, eta)) - (a ** 2 + b ** 2) / 8

def metropolis(langkah, T=5000, seed=0):            #<1>
    r = np.random.default_rng(seed)
    th, lp, jejak, terima = np.zeros(2), log_post(np.zeros(2)), [], 0
    for _ in range(T):
        usul = th + langkah * r.normal(size=2)      # usulan simetris
        lp_u = log_post(usul)
        if np.log(r.uniform()) < lp_u - lp:         # aturan penerimaan  #<2>
            th, lp, terima = usul, lp_u, terima + 1
        jejak.append(th)
    return np.array(jejak), terima / T

def ess(z):                                         # ukuran sampel efektif  #<3>
    z = z - z.mean(); n = len(z)
    rho = np.correlate(z, z, "full")[n - 1:] / (z.var() * n)
    k = np.argmax(rho < 0.05)                       # potong saat autokorelasi kecil
    return n / (1 + 2 * rho[1:k].sum())

for langkah in [0.005, 0.08, 1.0]:
    J, acc = metropolis(langkah)
    b = J[1000:, 1]                                 # buang 1000 sampel awal  #<4>
    print(f"langkah {langkah:5.3f}: diterima {acc:5.1%}, E[b] = {b.mean():+.3f},"
          f" sd[b] = {b.std():.3f}, ESS {ess(b):4.0f}")
