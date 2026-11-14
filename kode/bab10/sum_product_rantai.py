import itertools, time
import numpy as np

# Rantai N unit konveyor: x_n = 1 bila unit n bermasalah (ilustrasi).
# Unit bertetangga cenderung sama kondisinya; sensor murah memberi bukti lemah.
rng = np.random.default_rng(0)
N = 16
psi = np.array([[0.9, 0.1], [0.1, 0.9]])            # potensial pasangan  #<1>
phi = np.c_[rng.uniform(0.5, 0.9, N), rng.uniform(0.1, 0.5, N)]   # bukti sensor
phi[[5, 7, 8], 1] *= 20                             # sensor unit 6, 8, 9 panas

def brute_force():                                  # 2^N konfigurasi
    marg, terbaik = np.zeros((N, 2)), (0, None)
    for x in itertools.product((0, 1), repeat=N):
        p = np.prod(phi[np.arange(N), x]) * np.prod(psi[x[:-1], x[1:]])
        marg[np.arange(N), x] += p
        if p > terbaik[0]: terbaik = (p, x)
    return marg / marg.sum(1, keepdims=True), np.array(terbaik[1])

def sum_product():                                  # pesan maju dan mundur  #<2>
    maju, mundur = np.ones((N, 2)), np.ones((N, 2))
    for n in range(1, N):
        m = (maju[n-1] * phi[n-1]) @ psi; maju[n] = m / m.sum()
    for n in range(N - 2, -1, -1):
        m = psi @ (mundur[n+1] * phi[n+1]); mundur[n] = m / m.sum()
    b = maju * phi * mundur
    return b / b.sum(1, keepdims=True)

def max_sum():                                      # Viterbi dalam log  #<3>
    lpsi, lphi = np.log(psi), np.log(phi)
    skor, asal = lphi[0].copy(), np.zeros((N, 2), int)
    for n in range(1, N):
        cal = skor[:, None] + lpsi
        asal[n] = cal.argmax(0); skor = cal.max(0) + lphi[n]
    x = [int(skor.argmax())]
    for n in range(N - 1, 0, -1): x.append(asal[n, x[-1]])
    return np.array(x[::-1])

t0 = time.perf_counter(); m_bf, x_bf = brute_force(); t_bf = time.perf_counter() - t0
t0 = time.perf_counter(); m_sp = sum_product(); x_ms = max_sum(); t_sp = time.perf_counter() - t0
print("p(bermasalah):", " ".join(f"{v:.2f}" for v in m_sp[:8, 1]))
print("              ", " ".join(f"{v:.2f}" for v in m_sp[8:, 1]))
print(f"selisih maks. marginal brute force vs sum-product: {np.abs(m_bf - m_sp).max():.1e}")
print("MAP max-sum :", "".join(map(str, x_ms)), "| sama dengan brute force:", np.array_equal(x_bf, x_ms))
print(f"waktu: brute force {t_bf:.2f} s, pengiriman pesan {t_sp*1e3:.2f} ms")
