import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score as ari

# Titik panas (hotspot) satelit sintetis di wilayah 100 km x 100 km (ilustrasi):
# dua kawasan gambut berbentuk lengkung, satu kawasan bulat, dan titik sporadis.
rng = np.random.default_rng(7)
t = rng.uniform(0, np.pi, 260)
lengkung1 = np.c_[30 + 25 * np.cos(t), 45 + 25 * np.sin(t)] + rng.normal(0, 1.8, (260, 2))
t = rng.uniform(0, np.pi, 220)
lengkung2 = np.c_[55 - 25 * np.cos(t), 50 - 25 * np.sin(t)] + rng.normal(0, 1.8, (220, 2))
bulat = rng.normal([82, 82], 3.5, (120, 2))
sporadis = rng.uniform(0, 100, (60, 2))                    # pembakaran kecil, derau
X = np.vstack([lengkung1, lengkung2, bulat, sporadis])
z = np.repeat([0, 1, 2, -1], [260, 220, 120, 60])

db = DBSCAN(eps=5.0, min_samples=8).fit(X)                 #<1>
l = db.labels_
print(f"DBSCAN : {l.max() + 1} klaster, {np.sum(l == -1)} titik derau")
print("         ukuran klaster:", np.bincount(l[l >= 0]))
inti = np.zeros(len(X), bool); inti[db.core_sample_indices_] = True   #<2>
print(f"         titik inti {inti.sum()}, titik tepi {np.sum(~inti & (l >= 0))}")
km = KMeans(3, n_init=10, random_state=0).fit_predict(X)
m = z >= 0                                                 # bandingkan pada titik non-derau
print(f"ARI (tanpa titik sporadis): DBSCAN {ari(z[m], l[m]):.2f}, K-means {ari(z[m], km[m]):.2f}")
for eps in [2.0, 3.5, 5.0, 8.0, 12.0]:                     # kepekaan terhadap eps  #<3>
    l2 = DBSCAN(eps=eps, min_samples=8).fit_predict(X)
    print(f"eps = {eps:4.1f} km: {l2.max() + 1:2d} klaster, {np.sum(l2 == -1):3d} derau")
