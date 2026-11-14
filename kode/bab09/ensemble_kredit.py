import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_umkm import buat_data

X, y = buat_data()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
auc = lambda m: roc_auc_score(yu, m.predict_proba(Xu)[:, 1])

pohon = DecisionTreeClassifier(min_samples_leaf=5, random_state=0).fit(Xl, yl)
print(f"AUC uji satu pohon dalam: {auc(pohon):.3f}")
print("  B | bagging (uji) | random forest (uji) | random forest (OOB)")
for B in (25, 100, 500):                           # banyaknya pohon  #<1>
    bag = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=B,
                            random_state=0, n_jobs=-1).fit(Xl, yl)
    rf = RandomForestClassifier(n_estimators=B, min_samples_leaf=5, max_features="sqrt",
                                oob_score=True, random_state=0, n_jobs=-1).fit(Xl, yl)  #<2>
    oob = roc_auc_score(yl, rf.oob_decision_function_[:, 1])
    print(f"{B:3d} | {auc(bag):13.3f} | {auc(rf):19.3f} | {oob:19.3f}")                                         #<3>
