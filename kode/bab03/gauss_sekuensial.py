import numpy as np

# Ilustrasi: rerata tinggi anak usia 24 bulan di satu kecamatan
rng = np.random.default_rng(3)
sigma = 3.5                          # simpangan baku diketahui (cm)
x = rng.normal(84.0, sigma, size=50) # "data posyandu", rerata sebenarnya 84

m, s2 = 87.0, 2.0**2                 # prior N(87, 2^2) dari kurva rujukan  #<1>
for n, xn in enumerate(x, start=1):
    s2_baru = 1 / (1/s2 + 1/sigma**2)            # presisi bertambah  #<2>
    m = s2_baru * (m/s2 + xn/sigma**2)           # rerata tertimbang presisi
    s2 = s2_baru
    if n in (1, 2, 5, 10, 50):
        print(f"N={n:2d} | ML = {x[:n].mean():6.2f} | "
              f"posterior = {m:6.2f} +- {np.sqrt(s2):.2f}")

# Sekuensial = sekaligus: rumus batch untuk N = 50  #<3>
N = len(x)
s2_N = 1 / (1/2.0**2 + N/sigma**2)
m_N = s2_N * (87.0/2.0**2 + x.sum()/sigma**2)
print(f"batch N=50          posterior = {m_N:6.2f} +- {np.sqrt(s2_N):.2f}")
