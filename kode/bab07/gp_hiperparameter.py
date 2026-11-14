import numpy as np
from scipy.optimize import minimize
from data_pm25 import jam

x, t, _, _ = jam()
t = t - t.mean()

def log_marginal(log_theta):                      # ln p(t | theta)  #<1>
    amp, ell, sd = np.exp(log_theta)
    C = amp**2 * np.exp(-(x[:, None] - x[None, :])**2 / (2 * ell**2)) + sd**2 * np.eye(len(x))
    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))
    kecocokan = -0.5 * t @ alpha                   # seberapa cocok dengan data
    kompleksitas = -np.sum(np.log(np.diag(L)))     # -1/2 ln|C|  #<2>
    return kecocokan + kompleksitas - 0.5 * len(x) * np.log(2 * np.pi), kecocokan, kompleksitas

print(" skala (jam)  ln p(t)   kecocokan  kompleksitas")
for ell in (0.5, 2, 4, 8, 30):
    lp, a, b = log_marginal(np.log([15, ell, 4]))
    print(f"{ell:11.1f} {lp:9.1f} {a:10.1f} {b:13.1f}")

hasil = minimize(lambda z: -log_marginal(z)[0], np.log([10, 3, 2]))   #<3>
amp, ell, sd = np.exp(hasil.x)
print(f"optimum: amplitudo {amp:.1f}, skala {ell:.1f} jam, sd derau {sd:.1f} "
      "(sebenarnya 4.0)")
print(f"ln p(t) di optimum = {-hasil.fun:.1f}")
