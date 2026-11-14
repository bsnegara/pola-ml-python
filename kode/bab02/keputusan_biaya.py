import numpy as np
from scipy.stats import norm

# Model ilustrasi: jumlah trombosit (ribu per mikroliter) pada pasien demam
prior = 0.05
dbd, lain = norm(100, 40), norm(240, 40)        # p(x | kelas)  #<1>

def posterior(x):                               # p(DBD | x)
    a = dbd.pdf(x) * prior
    return a / (a + lain.pdf(x) * (1 - prior))

rng = np.random.default_rng(1)
n = 200_000
y = rng.random(n) < prior                       # True = DBD
x = np.where(y, dbd.rvs(n, random_state=rng), lain.rvs(n, random_state=rng))
p = posterior(x)

L_lewat, L_rujuk = 20, 1     # biaya: DBD terlewat vs rujukan yang tidak perlu  #<2>
def laporan(nama, rujuk, mask=None):
    m = np.ones(n, bool) if mask is None else mask
    salah = (rujuk != y)[m].mean()
    lewat = (y & ~rujuk)[m].sum()
    biaya = (L_lewat * (y & ~rujuk) + L_rujuk * (~y & rujuk))[m].mean()
    print(f"{nama:28s} salah={salah:6.2%}  DBD terlewat={lewat:5d}  biaya={biaya:.3f}")

laporan("A: p > 0,5 (galat minimum)", p > 0.5)
laporan("B: p > 1/21 (biaya minimum)", p > L_rujuk / (L_rujuk + L_lewat))  #<3>
yakin = np.maximum(p, 1 - p) >= 0.95                                       #<4>
laporan("C: B + tolak bila ragu", p > 1 / 21, yakin)
print(f"   ditolak (perlu tes NS1): {1 - yakin.mean():.1%} pasien")
for t in [80, 120, 150, 180]:
    print(f"   trombosit {t:3d} -> p(DBD | x) = {posterior(t):.3f}")
