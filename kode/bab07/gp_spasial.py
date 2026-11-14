import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from data_pm25 import sensor, medan

X, t = sensor(n=40)                                # 40 sensor murah
g = np.linspace(0, 20, 81)
G = np.array([(a, b) for b in g for a in g])       # grid peta 81 x 81
f_benar = medan(G[:, 0], G[:, 1])

kern = ConstantKernel(100) * RBF(length_scale=3.0) + WhiteKernel(5.0)   #<1>
gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=5,
                              random_state=0).fit(X, t)                   #<2>
print("kernel hasil optimasi:\n ", gp.kernel_)
m, s = gp.predict(G, return_std=True)              # rerata dan sd di setiap sel  #<3>

def idw(G, X, t, p=2):                             # pembanding: inverse distance weighting
    d = np.linalg.norm(G[:, None] - X[None], axis=2) + 1e-9
    w = d**-p
    return (w @ t) / w.sum(1)

for nama, pred in [("rata-rata sensor", np.full(len(G), t.mean())),
                   ("IDW", idw(G, X, t)), ("proses Gaussian", m)]:
    print(f"{nama:17s}: RMSE peta {np.sqrt(np.mean((pred - f_benar)**2)):5.2f} ug/m3")
s_f = np.sqrt(np.maximum(s**2 - gp.kernel_.k2.noise_level * np.var(t), 0))  # tanpa derau
print(f"medan sebenarnya di dalam +-2sd: {np.mean(np.abs(f_benar - m) < 2 * s_f):.0%} sel")
