import numpy as np
from scipy.stats import norm

# Ilustrasi: tinggi anak 24 bulan, campuran 25% stunting dan 75% normal
rng = np.random.default_rng(7)
def sampel(n):
    stunting = rng.random(n) < 0.25
    return np.where(stunting, rng.normal(80, 3, n), rng.normal(87, 3.2, n))
x_latih, x_val = sampel(200), sampel(1000)

def kde(x, data, h):                                  #<1>
    return norm.pdf((x[:, None] - data[None, :]) / h).mean(axis=1) / h

for h in (0.3, 0.8, 1.5, 3.0, 6.0):
    ll = np.log(kde(x_val, x_latih, h)).mean()        # log-likelihood validasi  #<2>
    print(f"h = {h:3.1f} cm      | log-likelihood validasi rata-rata = {ll:.3f}")

h_silverman = 1.06 * x_latih.std() * len(x_latih) ** (-1/5)   #<3>
print(f"aturan Silverman  | h = {h_silverman:.2f} cm")

g = norm(x_latih.mean(), x_latih.std())               # pembanding: satu Gaussian
print(f"satu Gaussian     | log-likelihood validasi rata-rata = "
      f"{g.logpdf(x_val).mean():.3f}")
