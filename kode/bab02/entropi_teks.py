import numpy as np
from collections import Counter

teks = ("Bahwa sesungguhnya kemerdekaan itu ialah hak segala bangsa dan oleh "
        "sebab itu, maka penjajahan di atas dunia harus dihapuskan, karena tidak "
        "sesuai dengan perikemanusiaan dan perikeadilan.")   # Pembukaan UUD 1945  #<1>
huruf = [c for c in teks.lower() if c.isalpha()]
n = Counter(huruf)
p = np.array(list(n.values())) / len(huruf)

H = -np.sum(p * np.log2(p))                      # entropi (bit)  #<2>
print(f"{len(huruf)} huruf, {len(n)} jenis huruf berbeda")
print(f"entropi distribusi huruf : {H:.2f} bit per huruf")
print(f"batas atas log2({len(n)})      : {np.log2(len(n)):.2f} bit (semua huruf sama sering)")
print("lima huruf tersering    :", ", ".join(f"{h}={c}" for h, c in n.most_common(5)))

# Divergensi KL tidak simetris  #<3>
P = np.array([0.7, 0.2, 0.1])                    # misal: hujan, berawan, cerah (Januari)
Q = np.array([0.2, 0.3, 0.5])                    # (Agustus)
KL = lambda a, b: np.sum(a * np.log2(a / b))
print(f"KL(P||Q) = {KL(P, Q):.3f} bit,  KL(Q||P) = {KL(Q, P):.3f} bit")
