import numpy as np
from collections import Counter
from data_pos import buat_korpus, TAG

korpus = buat_korpus()
latih, uji = korpus[:2500], korpus[2500:]
kosakata = sorted({w for s in korpus for w, _ in s}); iw = {w: i for i, w in enumerate(kosakata)}
K, V = len(TAG), len(kosakata); it = {t: i for i, t in enumerate(TAG)}

pi, A, B = np.ones(K), np.ones((K, K)), np.ones((K, V))   # penghalusan +1  #<1>
for s in latih:
    t = [it[g] for _, g in s]; pi[t[0]] += 1
    for a, b in zip(t[:-1], t[1:]): A[a, b] += 1
    for (w, _), k in zip(s, t): B[k, iw[w]] += 1
pi, A, B = pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)
lpi, lA, lB = np.log(pi), np.log(A), np.log(B)

def viterbi(kata):                                         #<2>
    o = [iw[w] for w in kata]; d = lpi + lB[:, o[0]]; asal = []
    for x in o[1:]:
        c = d[:, None] + lA; asal.append(c.argmax(0)); d = c.max(0) + lB[:, x]
    jalur = [int(d.argmax())]
    for a in reversed(asal): jalur.append(int(a[jalur[-1]]))
    return [TAG[k] for k in jalur[::-1]]

def maju_mundur(kata):                                     #<3>
    o = [iw[w] for w in kata]; T = len(o)
    al, be = np.zeros((T, K)), np.ones((T, K))
    al[0] = pi * B[:, o[0]]
    for n in range(1, T): al[n] = (al[n-1] @ A) * B[:, o[n]]
    for n in range(T - 2, -1, -1): be[n] = A @ (B[:, o[n+1]] * be[n+1])
    return al * be / al[-1].sum(), al[-1].sum()

terbanyak = {}                                             # pembanding  #<4>
for w in kosakata:
    c = Counter(g for s in latih for x, g in s if x == w)
    terbanyak[w] = c.most_common(1)[0][0]
taksa = {w for w in kosakata if len({g for s in korpus for x, g in s if x == w}) > 1}
benar = {"hmm": [0, 0], "base": [0, 0]}
for s in uji:
    kata, gold = zip(*s); pred = viterbi(kata)
    for w, g, p in zip(kata, gold, pred):
        j = int(w in taksa)
        benar["hmm"][j] += p == g; benar["base"][j] += terbanyak[w] == g
n_taksa = sum(w in taksa for s in uji for w, _ in s); n = sum(len(s) for s in uji)
print("kata taksa:", sorted(taksa))
print(f"akurasi tag terbanyak : semua {(sum(benar['base']))/n:.3f},"
      f" kata taksa {benar['base'][1]/n_taksa:.3f}")
print(f"akurasi HMM + Viterbi : semua {(sum(benar['hmm']))/n:.3f},"
      f" kata taksa {benar['hmm'][1]/n_taksa:.3f}")
for kal in ["beliau baru masak tahu", "tahu itu baru masak", "saya tahu jalan itu"]:
    print(f"  {kal:24s} -> {' '.join(viterbi(kal.split()))}")
g, _ = maju_mundur("tahu itu baru masak".split())
print("p(tag | kalimat) untuk 'masak':",
      ", ".join(f"{t} {g[3, i]:.2f}" for i, t in enumerate(TAG) if g[3, i] > 0.01))
for kal in ["saya makan nasi enak", "nasi enak saya makan"]:   # peluang kalimat
    print(f"ln p('{kal}') = {np.log(maju_mundur(kal.split())[1]):.2f}")
