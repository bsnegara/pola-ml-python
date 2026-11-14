import numpy as np

rng = np.random.default_rng(7)
f = lambda x: 200 + 150 * np.cos(2 * np.pi * x)
M = 5                                            # cukup lentur untuk pola monsun
for N in [12, 50, 500]:
    var_ml = []
    for _ in range(2000):                        # ulangi eksperimen 2000 kali  #<1>
        x = rng.uniform(0, 1, N)
        t = f(x) + rng.normal(0, 40, N)
        w = np.polyfit(x, t, M)                  # w_ML = solusi kuadrat terkecil
        var_ml.append(np.mean((np.polyval(w, x) - t) ** 2))  # sigma^2_ML  #<2>
    v = np.mean(var_ml)
    v_kor = v * N / (N - (M + 1))                # koreksi derajat bebas  #<3>
    print(f"N={N:4d}   sigma_ML = {np.sqrt(v):4.1f} mm   "
          f"terkoreksi = {np.sqrt(v_kor):4.1f} mm   (sebenarnya 40)")
