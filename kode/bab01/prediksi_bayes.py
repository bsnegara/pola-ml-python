import numpy as np

rng = np.random.default_rng(3)
f = lambda x: 200 + 150 * np.cos(2 * np.pi * x)
x = rng.uniform(0, 1, 40)
x = x[(x < 5/12) | (x > 8/12)][:15]          # alat rusak Juni-Agustus  #<1>
t = f(x) + rng.normal(0, 40, x.size)

M, alpha, beta = 9, 1e-8, 1 / 40**2
Phi = lambda x: np.vander(np.atleast_1d(x), M + 1, increasing=True)
S = np.linalg.inv(alpha * np.eye(M + 1) + beta * Phi(x).T @ Phi(x))  #<2>
m = beta * S @ Phi(x).T @ t

def prediksi(x_baru):                        # distribusi prediktif  #<3>
    phi = Phi(x_baru)
    rerata = phi @ m
    ragam_model = np.sum(phi @ S * phi, axis=1)  # ketidakpastian tentang w
    return rerata, np.sqrt(ragam_model), np.sqrt(1 / beta + ragam_model)

print(f"titik latih: {x.size}, tanpa data antara Juni dan Agustus")
print("bulan     rerata  s_model  s_total  sebenarnya   (semua dalam mm)")
for nama, xb in [("Januari", 0.5/12), ("April", 3.5/12),
                 ("Juli", 6.5/12), ("Oktober", 9.5/12)]:
    r, s_m, s_t = prediksi(xb)
    print(f"{nama:8s} {r[0]:7.0f} {s_m[0]:8.0f} {s_t[0]:8.0f} {f(xb):11.0f}")
