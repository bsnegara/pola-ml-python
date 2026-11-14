import numpy as np

rng = np.random.default_rng(0)
print("   D   rasio jarak (maks-min)/min   volume di kulit 5%")
for D in [2, 10, 100, 1000]:
    X = rng.uniform(0, 1, (500, D))          # 500 titik acak dalam kubus satuan
    q = rng.uniform(0, 1, D)                 # satu titik kueri
    d = np.linalg.norm(X - q, axis=1)
    kontras = (d.max() - d.min()) / d.min()  #<1>
    kulit = 1 - 0.95**D                      # fraksi volume dekat permukaan  #<2>
    print(f"{D:5d}   {kontras:26.2f}   {kulit:18.3f}")
