import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from data_ulasan import buat_data

teks, y = buat_data()
for ng in [(1, 1), (1, 2)]:                                     # pembanding bag of words  #<1>
    cv = CountVectorizer(ngram_range=ng, token_pattern=r"\S+"); B = cv.fit_transform(teks)
    lr = LogisticRegression(max_iter=3000).fit(B[:5000], y[:5000])
    print(f"bag of words {ng}: akurasi uji {lr.score(B[5000:], y[5000:]):.3f}")

kosa = {w: i + 2 for i, w in enumerate(sorted({w for t in teks for w in t.split()}))}
L = 16                                                          # 0 = isian, 1 = [CLS]
def kode(t):
    ids = [1] + [kosa[w] for w in t.split()]
    return ids + [0] * (L - len(ids))
T = torch.tensor([kode(t) for t in teks]); Y = torch.tensor(y)

class Atensi(nn.Module):                                        # satu kepala atensi  #<2>
    def __init__(self, d):
        super().__init__(); self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
    def forward(self, h, topeng):
        s = self.q(h) @ self.k(h).transpose(1, 2) / h.shape[-1] ** 0.5
        a = torch.softmax(s.masked_fill(topeng[:, None, :], -1e9), -1)
        return a @ self.v(h), a

class Transformer(nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(L, d)   # posisi  #<3>
        self.at1, self.at2 = Atensi(d), Atensi(d)
        self.ff = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.keluar = nn.Linear(d, 2)
    def forward(self, x):
        topeng = x == 0
        h = self.emb(x) + self.pos(torch.arange(L))
        a, w1 = self.at1(h, topeng); h = self.n1(h + a)                  # residual  #<4>
        a, w2 = self.at2(h, topeng); h = self.n2(h + a)
        h = self.n3(h + self.ff(h))
        return self.keluar(h[:, 0]), w2                                   # vektor [CLS]

torch.manual_seed(0)
net = Transformer(len(kosa) + 2); opt = torch.optim.Adam(net.parameters(), 2e-3)
for ep in range(15):
    for idx in torch.randperm(5000, generator=torch.Generator().manual_seed(ep)).split(64):
        opt.zero_grad(); nn.functional.cross_entropy(net(T[idx])[0], Y[idx]).backward(); opt.step()
net.eval()
with torch.no_grad():
    akur = (net(T[5000:])[0].argmax(1) == Y[5000:]).float().mean()
    print(f"transformer 2 lapis atensi: akurasi uji {akur:.3f}")
    for t in ["pengirimannya tidak lambat", "kurirnya ramah tapi harganya mahal"]:
        logit, w = net(torch.tensor([kode(t)]))
        p = torch.softmax(logit, 1)[0, 1].item()
        kata = t.split(); bobot = w[0, 0, 1:len(kata) + 1]                # atensi [CLS]  #<5>
        print(f"'{t}': p(positif) = {p:.2f}")
        print("   atensi:", " ".join(f"{k}:{b:.2f}" for k, b in zip(kata, bobot)))
