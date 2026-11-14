import numpy as np

# Ilustrasi: tinggi (cm) dan berat (kg) balita usia sekitar 2 tahun
mu = np.array([85.0, 11.5])
s_t, s_b, rho = 3.5, 1.3, 0.7
Sigma = np.array([[s_t**2, rho*s_t*s_b], [rho*s_t*s_b, s_b**2]])  #<1>

# Distribusi berat bersyarat tinggi = 80 cm (rumus tertutup)  #<2>
t = 80.0
mu_bs = mu[1] + Sigma[1, 0] / Sigma[0, 0] * (t - mu[0])
var_bs = Sigma[1, 1] - Sigma[1, 0]**2 / Sigma[0, 0]
print(f"marginal berat      : {mu[1]:.2f} +- {np.sqrt(Sigma[1,1]):.2f} kg")
print(f"berat | tinggi=80cm : {mu_bs:.2f} +- {np.sqrt(var_bs):.2f} kg (rumus)")

# Periksa dengan simulasi: ambil sampel yang tingginya dekat 80 cm  #<3>
rng = np.random.default_rng(0)
X = rng.multivariate_normal(mu, Sigma, size=2_000_000)
dekat = np.abs(X[:, 0] - t) < 0.1
print(f"berat | tinggi~80cm : {X[dekat,1].mean():.2f} +- {X[dekat,1].std():.2f} kg "
      f"(simulasi, {dekat.sum()} sampel)")
