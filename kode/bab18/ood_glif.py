import sys, pathlib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bab16"))
from data_aksara import buat_data

X, y = buat_data(8000)
X = torch.tensor(X); y = torch.tensor(y)
kenal = y < 8                                            # glif 8 dan 9 tidak pernah dilatih  #<1>
Xl, yl = X[:6000][kenal[:6000]], y[:6000][kenal[:6000]]
Xu_in, Xu_ood = X[6000:][kenal[6000:]], X[6000:][~kenal[6000:]]
g = torch.Generator().manual_seed(1)                     # bercak acak
acak = torch.rand(500, 400, generator=g) * (torch.rand(500, 400, generator=g) < 0.2)

def latih(seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(400, 256), nn.ReLU(), nn.Linear(256, 8))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(15):
        for idx in torch.randperm(len(yl)).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(net(Xl[idx]), yl[idx]).backward(); opt.step()
    return net

ens = [latih(s) for s in range(5)]
H = lambda p: -(p * torch.log(p + 1e-12)).sum(-1)       # entropi
def skor(Z):
    with torch.no_grad():
        P = torch.stack([n(Z).softmax(-1) for n in ens])  # (anggota, n, kelas)
    pbar = P.mean(0)
    return {"1 - maks p (satu jaringan)": 1 - P[0].max(-1).values,
            "entropi total (ensemble)": H(pbar),                          #<2>
            "informasi mutual (epistemik)": H(pbar) - H(P).mean(0)}, pbar

s_in, p_in = skor(Xu_in)
print(f"akurasi uji glif 0-7: {(p_in.argmax(-1) == y[6000:][kenal[6000:]]).float().mean():.3f}")
print("skor                          AUROC glif 8-9  AUROC bercak acak")
for nama in s_in:
    a = [roc_auc_score(np.r_[np.zeros(len(s_in[nama])), np.ones(len(Z))],
                       np.r_[s_in[nama].numpy(), skor(Z)[0][nama].numpy()])
         for Z in (Xu_ood, acak)]
    print(f"{nama:30s} {a[0]:10.3f} {a[1]:14.3f}")
s_ood, p_ood = skor(Xu_ood)
print(f"rerata maks p ensemble: kenal {p_in.max(-1).values.mean():.2f},"
      f" glif baru {p_ood.max(-1).values.mean():.2f}")
