import numpy as np
import torch
from torch import nn

# Data yang sama dengan Bab 1: 15 pengamatan, alat rusak Juni-Agustus.
rng = np.random.default_rng(3)
f = lambda x: 200 + 150 * np.cos(2 * np.pi * x)
x = rng.uniform(0, 1, 40)
x = x[(x < 5/12) | (x > 8/12)][:15]
t = f(x) + rng.normal(0, 40, x.size)
X = torch.tensor((x[:, None] - 0.5) * 4, dtype=torch.float32)       # skala masukan
T = torch.tensor(((t - 200) / 100)[:, None], dtype=torch.float32)   # skala target
mm = lambda a: a * 100 + 200                                         # kembali ke mm

def latih(idx, p_drop=0.0, seed=0, iterasi=1500):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Dropout(p_drop),
                        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-3)   # ~ prior Gaussian
    for _ in range(iterasi):
        opt.zero_grad(); ((net(X[idx]) - T[idx]) ** 2).mean().backward(); opt.step()
    return net

semua = np.arange(15)
tunggal = latih(semua)                                    # satu jaringan biasa
mc = latih(semua, p_drop=0.2)                             # MC dropout  #<1>
boot = [rng.choice(15, 15) for _ in range(20)]            # ensemble + bootstrap  #<2>
ens = [latih(b, seed=s) for s, b in enumerate(boot)]

xb = torch.tensor([[0.5 / 12], [3.5 / 12], [6.5 / 12], [9.5 / 12]])
xb = (xb - 0.5) * 4
with torch.no_grad():
    r1 = mm(tunggal(xb)[:, 0].numpy())
    mc.train()                                            # dropout tetap aktif saat prediksi
    S_mc = mm(torch.stack([mc(xb)[:, 0] for _ in range(500)]).numpy())
    S_en = mm(torch.stack([n(xb)[:, 0] for n in ens]).numpy())
    P = mm(torch.stack([n(X)[:, 0] for n in ens]).numpy())
oob = [np.mean([P[k, i] for k, b in enumerate(boot) if i not in b]) for i in range(15)]
s_alea = np.sqrt(np.nanmean((np.array(oob) - t) ** 2))    # derau dari galat out-of-bag  #<3>

print(f"taksiran derau aleatorik (out-of-bag): {s_alea:.0f} mm (sebenarnya 40 mm)")
print("bulan    sebenarnya | tunggal | MC dropout | ensemble 20 jaringan")
print("                    | rerata  | rerata sd  | rerata s_epi s_tot")
for i, b in enumerate(["Januari", "April", "Juli", "Oktober"]):
    s_tot = np.sqrt(S_en[:, i].var() + s_alea ** 2)       #<4>
    print(f"{b:8s} {f(1 / 24 + i / 4):9.0f}  | {r1[i]:6.0f}  | {S_mc[:, i].mean():6.0f}"
          f" {S_mc[:, i].std():3.0f} | {S_en[:, i].mean():6.0f} {S_en[:, i].std():5.0f} {s_tot:5.0f}")

xg = np.linspace(0, 1, 121)                               # untuk gambar (tidak dicetak)
with torch.no_grad():
    G = mm(torch.stack([n(torch.tensor((xg[:, None] - .5) * 4, dtype=torch.float32))[:, 0]
                        for n in ens]).numpy())
np.savetxt("_kurva_hujan.txt", np.c_[xg, f(xg), G.mean(0), G.std(0),
           np.sqrt(G.var(0) + s_alea ** 2)], fmt="%.4f")
np.savetxt("_titik_hujan.txt", np.c_[x, t], fmt="%.4f")
