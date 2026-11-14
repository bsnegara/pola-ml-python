import numpy as np
from sklearn.linear_model import LogisticRegression
from data_sms import buat_korpus, kantong_kata

teks, _, jenis = buat_korpus(n_normal=1500, n_penipuan=1500, seed=1)
baru = ["Selamat paket anda pemenang, klik foto resi",          # campuran  #<1>
        "Kak pinjam dana dulu ya, nanti aku transfer balik",
        "Besok rapat jam 9, jangan lupa bawa laptop"]
semua = list(teks) + baru
latih = np.r_[np.ones(len(teks), bool), np.zeros(len(baru), bool)]
X, _ = kantong_kata(semua, latih)

model = LogisticRegression(C=1.0, max_iter=2000)    # softmax dengan prior L2  #<2>
model.fit(X[latih], jenis)
P = model.predict_proba(X[~latih])                  # satu baris = 6 peluang  #<3>
print(" " * 9 + "".join(f"{k:>9s}" for k in model.classes_))
for i, p in enumerate(P, start=1):
    print(f"pesan {i}: " + "".join(f"{v:9.2f}" for v in p))
