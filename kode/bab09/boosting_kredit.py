import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from data_umkm import buat_data, FITUR

X, y = buat_data()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
kategori = [FITUR.index("sektor")]                 # sektor diperlakukan kategoris  #<1>

print(" laju belajar | pohon dipakai | AUC uji | log-loss uji")
for lr in (1.0, 0.3, 0.1, 0.03):                   # shrinkage (shrinkage)  #<2>
    gb = HistGradientBoostingClassifier(learning_rate=lr, max_iter=2000, max_depth=3,
                                        early_stopping=True, validation_fraction=0.15,
                                        n_iter_no_change=30, categorical_features=kategori,
                                        random_state=0).fit(Xl, yl)            #<3>
    p = gb.predict_proba(Xu)[:, 1]
    print(f"{lr:13.2f} | {gb.n_iter_:13d} | {roc_auc_score(yu, p):7.3f} | {log_loss(yu, p):12.4f}")
