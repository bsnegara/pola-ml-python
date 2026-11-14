import torch
from torch import nn
from data_daun import buat_data, KELAS

X, y = buat_data()
X, y = torch.tensor(X), torch.tensor(y)
Xl, yl, Xu, yu = X[:800], y[:800], X[3000:], y[3000:]       # data latih sedikit

def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 128), nn.ReLU(),
                         nn.Linear(128, 4))

def cnn():                                                   #<1>
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 16 x 16
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 8 x 8
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(),                        # maks. global  #<2>
        nn.Linear(32, 4))

def latih(buat, epoch=40):
    torch.manual_seed(0); net = buat()
    opt = torch.optim.Adam(net.parameters(), 2e-3); f = nn.CrossEntropyLoss()
    for ep in range(epoch):
        for idx in torch.randperm(800, generator=torch.Generator().manual_seed(ep)).split(32):
            opt.zero_grad(); f(net(Xl[idx]), yl[idx]).backward(); opt.step()
    return net

for nama, buat in [("MLP", mlp), ("CNN", cnn)]:
    net = latih(buat).eval()
    with torch.no_grad():
        p = net(Xu).argmax(1)
        geser = net(torch.roll(Xu, shifts=(8, 8), dims=(2, 3))).argmax(1)   # citra digeser  #<3>
    n_par = sum(w.numel() for w in net.parameters())
    print(f"{nama}: {n_par:6d} parameter, akurasi uji {(p == yu).float().mean():.3f},"
          f" uji digeser {(geser == yu).float().mean():.3f}")
    per_kelas = [(p[yu == k] == k).float().mean().item() for k in range(4)]
    print("     per kelas:", ", ".join(f"{n} {a:.2f}" for n, a in zip(KELAS, per_kelas)))
