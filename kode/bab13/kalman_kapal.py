import numpy as np

# Kapal di selat (ilustrasi): keadaan z = [x, y, vx, vy] dalam km dan km/menit.
rng = np.random.default_rng(2)
T, dt = 120, 1.0
A = np.eye(4); A[0, 2] = A[1, 3] = dt                   # kecepatan konstan  #<1>
G = np.eye(4)[:2]                                       # hanya posisi teramati
Q = np.diag([0, 0, 1e-4, 1e-4])                         # percepatan acak kecil
R = 0.4 ** 2 * np.eye(2)                                # derau posisi 0,4 km
z = np.zeros((T, 4)); z[0] = [0, 0, 0.30, 0.05]
for n in range(1, T):
    z[n] = A @ z[n-1] + rng.multivariate_normal(np.zeros(4), Q)
    if n == 78: z[n, 2:] = [0.10, 0.28]                 # berbelok saat sinyal hilang
x = z[:, :2] + rng.normal(0, 0.4, (T, 2))
ada = np.ones(T, bool); ada[70:90] = False              # sinyal hilang 20 menit

def kalman(x, ada):                                     #<2>
    m, P = np.r_[x[0], 0, 0], np.diag([0.16, 0.16, 1, 1])
    M, S, Mp, Sp = [], [], [], []
    for n in range(T):
        if n > 0:                                       # prediksi
            m, P = A @ m, A @ P @ A.T + Q
        Mp.append(m); Sp.append(P)
        if ada[n]:                                      # koreksi dengan pengamatan
            Kg = P @ G.T @ np.linalg.inv(G @ P @ G.T + R)
            m, P = m + Kg @ (x[n] - G @ m), (np.eye(4) - Kg @ G) @ P
        M.append(m); S.append(P)
    return map(np.array, (M, S, Mp, Sp))

def rts(M, S, Mp, Sp):                                  # penghalus  #<3>
    Mh, Sh = M.copy(), S.copy()
    for n in range(T - 2, -1, -1):
        J = S[n] @ A.T @ np.linalg.inv(Sp[n+1])
        Mh[n] = M[n] + J @ (Mh[n+1] - Mp[n+1])
        Sh[n] = S[n] + J @ (Sh[n+1] - Sp[n+1]) @ J.T
    return Mh, Sh

M, S, Mp, Sp = kalman(x, ada); Mh, Sh = rts(M, S, Mp, Sp)
galat = lambda e, m: np.sqrt(np.mean(np.sum((e[m, :2] - z[m, :2]) ** 2, 1)))
for nama, e in [("pengamatan", np.c_[x, x]), ("filter", M), ("penghalus", Mh)]:
    print(f"RMSE {nama:10s}: saat ada sinyal {galat(e, ada):.3f} km"
          + ("" if nama == "pengamatan" else f", saat hilang {galat(e, ~ada):.3f} km"))
sd = lambda S_, n: np.sqrt(S_[n, 0, 0])
print(f"simp. baku posisi-x (filter): menit 69 {sd(S, 69):.2f}, menit 89 {sd(S, 89):.2f},"
      f" menit 90 {sd(S, 90):.2f}")
