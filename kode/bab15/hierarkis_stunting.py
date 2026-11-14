import numpy as np
import pymc as pm
import arviz as az
from data_stunting import buat_data

y, n, theta_benar = buat_data()
J = len(y)

with pm.Model() as model:                                  # model hierarkis  #<1>
    mu = pm.Normal("mu", -1.3, 1.0)                        # logit rerata provinsi
    sigma = pm.HalfNormal("sigma", 0.5)                    # sebaran antarkabupaten
    eta = pm.Normal("eta", 0, 1, shape=J)                  # parameterisasi tak terpusat  #<2>
    theta = pm.Deterministic("theta", pm.math.invlogit(mu + sigma * eta))
    pm.Binomial("y", n=n, p=theta, observed=y)
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=0,
                      progressbar=False)                   # NUTS  #<3>

ringkas = az.summary(idata, var_names=["mu", "sigma"])     #<4>
print("         mean     sd ess_bulk r_hat")
for v, r in ringkas.iterrows():
    print(f"{v:5s} {r['mean']:7.3f} {r['sd']:6.3f} {r['ess_bulk']:8.0f} {r['r_hat']:5.2f}")
print("divergensi:", int(idata.sample_stats["diverging"].sum()))
th = idata.posterior["theta"].mean(("chain", "draw")).values
mentah = y / n
kecil = n < 50
print("galat absolut rata-rata   semua   n < 50   n >= 50")
for nama, est in [("y/n saja ", mentah), ("gabungan ", np.full(J, y.sum() / n.sum())),
                  ("hierarkis", th)]:                      # tiga strategi  #<5>
    e = np.abs(est - theta_benar)
    print(f"  {nama}               {e.mean():.4f}   {e[kecil].mean():.4f}   {e[~kecil].mean():.4f}")
for j in np.argsort(n)[[0, 1, -1]]:
    print(f"  kab. #{j:2d}: n = {n[j]:3d}, y/n = {mentah[j]:.3f},"
          f" hierarkis = {th[j]:.3f}, sebenarnya = {theta_benar[j]:.3f}")
