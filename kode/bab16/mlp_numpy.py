import numpy as np
from data_aksara import buat_data

X, y = buat_data()
Xl, yl, Xu, yu = X[:5000], y[:5000], X[5000:], y[5000:]
rng = np.random.default_rng(0)
D, H, K = 400, 64, 10
P = {"W1": rng.normal(0, np.sqrt(2 / D), (D, H)), "b1": np.zeros(H),   # inisialisasi He  #<1>
     "W2": rng.normal(0, np.sqrt(1 / H), (H, K)), "b2": np.zeros(K)}

def maju(P, X):
    a1 = X @ P["W1"] + P["b1"]; z1 = np.maximum(a1, 0)          # ReLU
    a2 = z1 @ P["W2"] + P["b2"]
    a2 = a2 - a2.max(1, keepdims=True)                            # softmax stabil
    p = np.exp(a2); p /= p.sum(1, keepdims=True)
    return a1, z1, p

def loss_dan_gradien(P, X, y):
    a1, z1, p = maju(P, X); N = len(y)
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()             # -log-likelihood  #<2>
    d2 = p.copy(); d2[np.arange(N), y] -= 1; d2 /= N              # delta keluaran = p - t  #<3>
    d1 = (d2 @ P["W2"].T) * (a1 > 0)                              # propagasi balik  #<4>
    return loss, {"W2": z1.T @ d2, "b2": d2.sum(0), "W1": X.T @ d1, "b1": d1.sum(0)}

# periksa gradien dengan beda hingga pada satu bobot  #<5>
_, g = loss_dan_gradien(P, Xl[:50], yl[:50]); e = 1e-5
Pp = {k: v.copy() for k, v in P.items()}; Pp["W1"][10, 3] += e
Pm = {k: v.copy() for k, v in P.items()}; Pm["W1"][10, 3] -= e
num = (loss_dan_gradien(Pp, Xl[:50], yl[:50])[0] - loss_dan_gradien(Pm, Xl[:50], yl[:50])[0]) / (2 * e)
print(f"gradien backprop {g['W1'][10, 3]:+.6e}, beda hingga {num:+.6e}")

v = {k: np.zeros_like(w) for k, w in P.items()}
for ep in range(15):                                              # SGD + momentum  #<6>
    for idx in np.array_split(rng.permutation(5000), 5000 // 64):
        loss, g = loss_dan_gradien(P, Xl[idx], yl[idx])
        for k in P:
            v[k] = 0.9 * v[k] - 0.05 * g[k]; P[k] += v[k]
    if ep % 3 == 2:
        akur = (maju(P, Xu)[2].argmax(1) == yu).mean()
        print(f"epoch {ep + 1:2d}: loss latih {loss_dan_gradien(P, Xl, yl)[0]:.3f}, akurasi uji {akur:.3f}")
from sklearn.linear_model import LogisticRegression                # pembanding linear
lin = LogisticRegression(max_iter=2000).fit(Xl, yl)
print(f"regresi logistik (tanpa lapisan tersembunyi): akurasi uji {lin.score(Xu, yu):.3f}")
