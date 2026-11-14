import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from data_transaksi import buat_transaksi

X, y, hari = buat_transaksi()
latih, uji = hari < 120, hari >= 120
model = {"logistik": LogisticRegression(max_iter=1000),
         "gradient boosting": HistGradientBoostingClassifier(random_state=0)}  #<1>
p = {k: m.fit(X[latih], y[latih]).predict_proba(X[uji])[:, 1] for k, m in model.items()}
for k, pk in p.items():
    print(f"{k:18s} ROC-AUC {roc_auc_score(y[uji], pk):.3f} | "
          f"AP (luas PR) {average_precision_score(y[uji], pk):.3f}")       #<2>

c_fp, c_fn = 20_000, 2_000_000        # biaya periksa manual vs kerugian lolos  #<3>
ambang_bayes = c_fp / (c_fp + c_fn)
for ambang in (0.5, ambang_bayes):
    pred = p["logistik"] > ambang
    fp = np.sum(pred & (y[uji] == 0)); fn = np.sum(~pred & (y[uji] == 1))
    print(f"ambang {ambang:.4f}: FP={fp:4d} FN={fn:3d} | "
          f"biaya total Rp{(fp * c_fp + fn * c_fn) / 1e6:6.1f} juta")
