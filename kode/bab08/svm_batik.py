import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from data_batik import buat_data, MOTIF

X, y = buat_data()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

pipa = make_pipeline(StandardScaler(), SVC(kernel="rbf"))                 #<1>
kisi = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.03, 0.1, 0.3]}
cari = GridSearchCV(pipa, kisi, cv=5).fit(Xl, yl)                        #<2>
print(f"terbaik: {cari.best_params_} | akurasi CV {cari.best_score_:.3f}")
svm = cari.best_estimator_
print(f"SVM RBF         : akurasi uji {svm.score(Xu, yu):.3f} | "
      f"support vector {svm[-1].n_support_.sum()} dari {len(yl)}")    #<3>
for nama, m in [("SVM linear", SVC(kernel="linear")),
                ("regresi logistik", LogisticRegression(max_iter=3000))]:
    m = make_pipeline(StandardScaler(), m).fit(Xl, yl)
    print(f"{nama:16s}: akurasi uji {m.score(Xu, yu):.3f}")
print("kebingungan SVM RBF, baris = sebenarnya:", ", ".join(MOTIF))
print(confusion_matrix(yu, svm.predict(Xu)))
