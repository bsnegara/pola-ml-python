import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from data_harga import buat_data, fitur

t, y = buat_data()
latih, uji = t < 156, t >= 156
rng = np.random.default_rng(1)
acak = rng.normal(size=(t.size, 20)).cumsum(axis=0)   # 20 deret tak berkaitan  #<1>
X = np.column_stack([fitur(t, K=6)[:, 1:], acak])      # 14 fitur asli + 20 acak
nama_fitur = (["tren"] + [f"{f}{k}" for k in range(1, 7) for f in ("cos", "sin")]
              + ["Lebaran"] + [f"acak{j}" for j in range(20)])
skala = StandardScaler().fit(X[latih])                  #<2>
X = skala.transform(X)

print(f"{'model':16s} {'bukan nol':>9s} {'dari acak':>9s} {'RMSE uji':>9s}")
for nama, m in [("kuadrat terkecil", Ridge(alpha=1e-9)), ("ridge a=10", Ridge(alpha=10)),
                ("lasso a=0.05", Lasso(alpha=0.05, max_iter=50_000)),
                ("lasso a=0.4", Lasso(alpha=0.4, max_iter=50_000)),         #<3>
                ("lasso a=1.5", Lasso(alpha=1.5, max_iter=50_000))]:
    m.fit(X[latih], y[latih])
    nz = np.abs(m.coef_) > 1e-6
    rmse = np.sqrt(np.mean((m.predict(X[uji]) - y[uji])**2))
    print(f"{nama:16s} {nz.sum():9d} {nz[14:].sum():9d} {rmse:9.2f}")
    if nama == "lasso a=0.4":
        terpilih = [n for n, z in zip(nama_fitur, nz) if z]

print("lasso a=0.4 memilih:", ", ".join(terpilih))
