import time
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.linear_model import LogisticRegression

# Bagian 1: dua cincin konsentris (ilustrasi), tidak terpisah secara linear.
X, y = make_circles(400, factor=0.4, noise=0.06, random_state=0)
akur = lambda Y: LogisticRegression().fit(Y, y).score(Y, y)   # pemisah linear
print(f"PCA, 2 komponen       : akurasi {akur(PCA(2).fit_transform(X)):.2f}")
K = KernelPCA(3, kernel="rbf", gamma=2.0).fit_transform(X)       #<1>
for k in [1, 2, 3]:
    print(f"kernel PCA, {k} komponen: akurasi {akur(K[:, :k]):.2f}")

# Bagian 2: 4 kelompok di permukaan melengkung, ditanam di 20 dimensi.
rng = np.random.default_rng(0)
t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, 1200); h = rng.uniform(0, 10, 1200)
g = np.digitize(t, np.linspace(1.5 * np.pi, 4.5 * np.pi, 5)[1:-1])   # 4 kelompok
R3 = np.c_[t * np.cos(t), h, t * np.sin(t)]                         # gulungan  #<2>
Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
X20 = np.c_[R3, np.zeros((1200, 17))] @ Q + rng.normal(0, 0.3, (1200, 20))
hasil = {}
for nama, m in [("PCA  ", PCA(2)), ("t-SNE", TSNE(2, perplexity=30, random_state=0))]:
    t0 = time.perf_counter(); Y = m.fit_transform(X20); dt = time.perf_counter() - t0
    tw = trustworthiness(X20, Y, n_neighbors=10)                    #<3>
    print(f"{nama}: trustworthiness 10-tetangga {tw:.3f}, waktu {dt:5.2f} s")
    hasil[nama.strip()] = Y
