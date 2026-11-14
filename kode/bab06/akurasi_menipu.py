import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from data_transaksi import buat_transaksi

X, y, hari = buat_transaksi()
latih, uji = hari < 120, hari >= 120               # 4 bulan latih, 2 bulan uji  #<1>
print(f"transaksi uji: {uji.sum()}, tidak wajar: {y[uji].sum()} ({y[uji].mean():.2%})")

def laporan(nama, pred):
    tn, fp, fn, tp = confusion_matrix(y[uji], pred, labels=[0, 1]).ravel()   #<2>
    akurasi = (tp + tn) / (tp + tn + fp + fn)
    presisi = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    f1 = 2 * presisi * recall / (presisi + recall) if tp else 0.0
    print(f"{nama:21s}{tp:4d}{fp:5d}{fn:4d} | {akurasi:.4f}  {presisi:7.3f} {recall:6.3f} {f1:5.3f}")

print(f"{'model':21s}  TP   FP  FN | akurasi presisi recall    F1")
laporan("selalu 'wajar'", np.zeros(uji.sum(), int))                    #<3>
model = LogisticRegression(max_iter=1000).fit(X[latih], y[latih])
p = model.predict_proba(X[uji])[:, 1]
laporan("logistik, ambang 0.5", (p > 0.5).astype(int))
laporan("logistik, ambang 0.1", (p > 0.1).astype(int))                 #<4>
