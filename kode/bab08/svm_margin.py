import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from data_batik import buat_data

X, y = buat_data()
pilih = (y == 0) | (y == 1)                         # parang vs kawung  #<1>
X2 = StandardScaler().fit_transform(X[pilih][:, [6, 7]])   # dua fitur spektrum
t = np.where(y[pilih] == 1, 1, -1)                  # label SVM: -1 dan +1

print("      C | vektor pend. | margin | slack | akurasi latih")
for C in (1000, 1, 0.01):                           # dari margin keras ke lunak  #<2>
    svm = SVC(kernel="linear", C=C).fit(X2, t)
    w = svm.coef_[0]
    lebar = 2 / np.linalg.norm(w)                   # lebar margin  #<3>
    xi = np.maximum(0, 1 - t * svm.decision_function(X2))   # variabel slack
    print(f"{C:7.2f} | {len(svm.support_):3d} dari {len(t)} | {lebar:6.2f} | "
          f"{xi.sum():5.1f} | {svm.score(X2, t):.3f}")
