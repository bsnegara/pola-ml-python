import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from data_transaksi import buat_transaksi

# (a) Kebocoran lewat seleksi fitur: 2000 fitur derau murni, label acak  #<1>
rng = np.random.default_rng(0)
Z, t = rng.normal(size=(200, 2000)), rng.integers(0, 2, 200)
bocor = SelectKBest(f_classif, k=20).fit(Z, t).transform(Z)      # pilih dulu, CV kemudian
auc_bocor = cross_val_score(LogisticRegression(), bocor, t, cv=5, scoring="roc_auc")
benar = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())   #<2>
auc_benar = cross_val_score(benar, Z, t, cv=5, scoring="roc_auc")
print(f"label acak | AUC CV bocor = {auc_bocor.mean():.2f} | AUC CV benar = {auc_benar.mean():.2f}")

# (b) Panah waktu: CV acak vs CV berurutan, dibanding kinerja masa depan  #<3>
X, y, hari = buat_transaksi()
latih, uji = hari < 120, hari >= 120
m = LogisticRegression(max_iter=1000)
for nama, cv in [("KFold acak", KFold(5, shuffle=True, random_state=0)),
                 ("TimeSeriesSplit", TimeSeriesSplit(5))]:
    s = cross_val_score(m, X[latih], y[latih], cv=cv, scoring="average_precision")
    print(f"{nama:16s}: AP taksiran CV = {s.mean():.3f}")
p = m.fit(X[latih], y[latih]).predict_proba(X[uji])[:, 1]
print(f"{'masa depan':16s}: AP data uji  = {average_precision_score(y[uji], p):.3f}")
