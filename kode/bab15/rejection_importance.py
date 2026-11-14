import numpy as np
from scipy.stats import norm, binom, beta
from scipy.special import logit

# Survei di satu kabupaten (sintetis): 41 dari 150 balita stunting.
y, n = 41, 150
log_prior = lambda t: norm.logpdf(logit(t), logit(0.21), 0.5) - np.log(t * (1 - t))
log_target = lambda t: log_prior(t) + binom.logpmf(y, n, t)    # tak ternormalisasi  #<1>

g = np.linspace(1e-4, 1 - 1e-4, 20001)                        # rujukan: grid
w = np.exp(log_target(g) - log_target(g).max()); w /= w.sum()
print(f"grid       : E[theta] = {np.sum(g * w):.4f}, p(>0.30) = {w[g > 0.3].sum():.4f}")

rng = np.random.default_rng(0)
M = np.exp(log_target(g).max())                              # batas atas, usulan seragam  #<2>
t = rng.uniform(0, 1, 200_000)
u = rng.uniform(0, M, t.size)
s = t[u < np.exp(log_target(t))]
print(f"rejection  : E[theta] = {s.mean():.4f}, p(>0.30) = {np.mean(s > 0.3):.4f},"
      f" terima {s.size / t.size:.1%}")

q = beta(8, 32)                                               # usulan importance  #<3>
t = q.rvs(20_000, random_state=1)
lw = log_target(t) - q.logpdf(t); w = np.exp(lw - lw.max()); w /= w.sum()
ess = 1 / np.sum(w ** 2)                                      # ukuran sampel efektif  #<4>
print(f"importance : E[theta] = {np.sum(w * t):.4f}, p(>0.30) = {w[t > 0.3].sum():.4f},"
      f" ESS {ess:.0f} dari {t.size}")
