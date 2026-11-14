import numpy as np

def citra_asli(h=48, w=64):                        # pola biner sederhana: +1 / -1
    y, x = np.mgrid[0:h, 0:w]
    z = -np.ones((h, w))
    for cx, cy, r in [(16, 20, 9), (30, 16, 7), (27, 31, 8), (48, 26, 12)]:
        z[(x - cx)**2 + (y - cy)**2 < r**2] = 1
    z[38:44, 6:58] = 1
    return z

rng = np.random.default_rng(0)
x_asli = citra_asli()
y = np.where(rng.random(x_asli.shape) < 0.10, -x_asli, x_asli)   # 10% piksel dibalik  #<1>

def icm(y, beta=1.0, eta=2.1, putaran=10):         # iterated conditional modes  #<2>
    x = y.copy()
    for k in range(putaran):
        berubah = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tetangga = sum(x[a, b] for a, b in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
                               if 0 <= a < x.shape[0] and 0 <= b < x.shape[1])
                baru = 1.0 if beta * tetangga + eta * y[i, j] > 0 else -1.0   #<3>
                berubah += baru != x[i, j]; x[i, j] = baru
        if berubah == 0:
            break
    return x, k + 1

galat = lambda z: np.mean(z != x_asli)
print(f"citra berderau   : {galat(y):.2%} piksel salah")
for beta in (0.0, 1.0, 3.0):
    x, k = icm(y, beta=beta)
    print(f"ICM, beta = {beta:.1f} : {galat(x):.2%} piksel salah ({k} putaran)")
