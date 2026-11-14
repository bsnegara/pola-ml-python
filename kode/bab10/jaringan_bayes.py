import itertools
import numpy as np

# Jaringan Bayes diagnosis mesin press (peluang ilustratif)  #<1>
# P: pelumasan kurang, L: beban berlebih, B: bantalan aus,
# G: getaran tinggi, S: suhu tinggi, A: bunyi abnormal
induk = {"P": [], "L": [], "B": ["P", "L"], "G": ["B"], "S": ["B", "L"], "A": ["G"]}
tabel = {  # p(variabel = 1 | nilai induk)
    "P": {(): 0.10}, "L": {(): 0.20},
    "B": {(0, 0): 0.02, (0, 1): 0.15, (1, 0): 0.30, (1, 1): 0.60},
    "G": {(0,): 0.05, (1,): 0.80},
    "S": {(0, 0): 0.05, (0, 1): 0.40, (1, 0): 0.50, (1, 1): 0.85},
    "A": {(0,): 0.10, (1,): 0.70}}
nama = list(induk)

def gabungan(x):                                   # aturan kali sepanjang graf  #<2>
    p = 1.0
    for v in nama:
        p1 = tabel[v][tuple(x[u] for u in induk[v])]
        p *= p1 if x[v] else 1 - p1
    return p

def tanya(target, bukti):                          # inferensi dengan enumerasi  #<3>
    num = den = 0.0
    for nilai in itertools.product((0, 1), repeat=len(nama)):
        x = dict(zip(nama, nilai))
        if all(x[k] == v for k, v in bukti.items()):
            p = gabungan(x); den += p
            num += p * x[target]
    return num / den

print(f"p(B=1)                 = {tanya('B', {}):.3f}")
print(f"p(B=1 | G=1)           = {tanya('B', {'G': 1}):.3f}")
print(f"p(B=1 | G=1, S=1)      = {tanya('B', {'G': 1, 'S': 1}):.3f}")
print(f"p(B=1 | A=1)           = {tanya('B', {'A': 1}):.3f}")
print(f"p(P=1 | B=1)           = {tanya('P', {'B': 1}):.3f}")      # explaining away  #<4>
print(f"p(P=1 | B=1, L=1)      = {tanya('P', {'B': 1, 'L': 1}):.3f}")
print(f"p(P=1 | L=1)           = {tanya('P', {'L': 1}):.3f}   (P dan L saling bebas)")
