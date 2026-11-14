import numpy as np
import torch
from torch import nn
from data_aksara import buat_data

X, y = buat_data()
Xl, yl = torch.tensor(X[:5000]), torch.tensor(y[:5000])
Xu, yu = torch.tensor(X[5000:]), torch.tensor(y[5000:])

def jaringan():                                            # sama dengan versi NumPy  #<1>
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(400, 64), nn.ReLU(), nn.Linear(64, 10))

hasil = {}
for nama, buat_opt in [("SGD", lambda p: torch.optim.SGD(p, lr=0.05)),       #<2>
                       ("momentum", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
                       ("Adam", lambda p: torch.optim.Adam(p, lr=1e-3))]:
    net = jaringan(); opt = buat_opt(net.parameters())
    loss_fn = nn.CrossEntropyLoss()                        # softmax + -log-likelihood
    kurva = []
    for ep in range(15):
        for idx in torch.randperm(5000, generator=torch.Generator().manual_seed(ep)).split(64):
            opt.zero_grad()
            loss = loss_fn(net(Xl[idx]), yl[idx])
            loss.backward()                                # backprop otomatis  #<3>
            opt.step()
        with torch.no_grad():
            kurva.append(loss_fn(net(Xl), yl).item())
            akur = (net(Xu).argmax(1) == yu).float().mean().item()
    hasil[nama] = kurva
    print(f"{nama:8s}: loss latih epoch 1 {kurva[0]:.3f}, epoch 15 {kurva[-1]:.3f},"
          f" akurasi uji {akur:.3f}")
np.savetxt("_kurva_opt.txt", np.c_[np.arange(1, 16), *hasil.values()], fmt="%.4f")
