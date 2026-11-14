import numpy as np

# Barisan harian 0 = kering, 1 = hujan (sintetis, ilustrasi musim peralihan).
rng = np.random.default_rng(0)
A_benar = np.array([[0.8, 0.2], [0.4, 0.6]])          # p(besok | hari ini)
x = [0]
for _ in range(729):
    x.append(rng.choice(2, p=A_benar[x[-1]]))
x = np.array(x)

C = np.zeros((2, 2))                                   # hitung transisi  #<1>
np.add.at(C, (x[:-1], x[1:]), 1)
A = C / C.sum(1, keepdims=True)
print("matriks transisi taksiran:\n", A.round(3))
w, V = np.linalg.eig(A.T)                              # distribusi stasioner  #<2>
pi = np.real(V[:, np.argmax(np.real(w))]); pi /= pi.sum()
print("distribusi stasioner:", pi.round(3), "| porsi hujan di data:", x.mean().round(3))
for n in [1, 2, 7]:                                    # peluang n hari ke depan  #<3>
    print(f"p(hujan {n} hari lagi | hari ini hujan) = {np.linalg.matrix_power(A, n)[1, 1]:.3f}")
