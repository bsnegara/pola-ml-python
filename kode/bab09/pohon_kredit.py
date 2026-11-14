import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_umkm import buat_data, FITUR

X, y = buat_data()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(f"gagal bayar: {y.mean():.1%} dari {len(y)} pinjaman")

for d in (1, 2, 4, 8, None):                       # kedalaman pohon  #<1>
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xl, yl)
    auc = [roc_auc_score(t, m.predict_proba(A)[:, 1]) for A, t in ((Xl, yl), (Xu, yu))]
    print(f"kedalaman {str(d):4s}: {m.get_n_leaves():4d} daun | AUC latih {auc[0]:.3f} | uji {auc[1]:.3f}")

pohon = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xl, yl)
print(export_text(pohon, feature_names=FITUR, show_weights=True, decimals=2))  #<2>
