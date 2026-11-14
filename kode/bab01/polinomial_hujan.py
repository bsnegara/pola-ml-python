import numpy as np

rng = np.random.default_rng(7)
def curah_hujan(x):                        # pola monsun "sebenarnya" (mm/bulan)
    return 200 + 150 * np.cos(2 * np.pi * x)  #<1>

def buat_data(n):
    x = rng.uniform(0, 1, n)               # waktu dalam setahun, 0 = 1 Januari
    t = curah_hujan(x) + rng.normal(0, 40, n)  # derau cuaca, sigma = 40 mm
    return x, t

x_latih, t_latih = buat_data(12)           # hanya 12 pengamatan  #<2>
x_uji, t_uji = buat_data(1000)

def rmse(w, x, t):
    return np.sqrt(np.mean((np.polyval(w, x) - t) ** 2))

print("orde M   RMSE latih (mm)   RMSE uji (mm)")
for M in [0, 1, 3, 9]:
    w = np.polyfit(x_latih, t_latih, M)    # kuadrat terkecil  #<3>
    print(f"{M:6d}   {rmse(w, x_latih, t_latih):15.1f}   {rmse(w, x_uji, t_uji):13.1f}")
