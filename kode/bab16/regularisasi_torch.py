import numpy as np
import torch
from torch import nn
from data_aksara import buat_data

X, y = buat_data()
Xl, yl = torch.tensor(X[:300]), torch.tensor(y[:300])          # data latih sedikit  #<1>
Xv, yv = torch.tensor(X[3000:4000]), torch.tensor(y[3000:4000])  # validasi
Xu, yu = torch.tensor(X[5000:]), torch.tensor(y[5000:])          # uji

def latih(wd=0.0, p_drop=0.0, epoch=300):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(400, 256), nn.ReLU(), nn.Dropout(p_drop),   #<2>
                        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p_drop),
                        nn.Linear(256, 10))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)        #<3>
    f = nn.CrossEntropyLoss(); kurva, terbaik = [], (np.inf, None, 0)
    for ep in range(epoch):
        net.train(); opt.zero_grad(); f(net(Xl), yl).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = f(net(Xv), yv).item(); kurva.append(lv)
            if lv < terbaik[0]:                                                #<4>
                terbaik = (lv, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    with torch.no_grad():
        ukur = lambda: (f(net(Xu), yu).item(), (net(Xu).argmax(1) == yu).float().mean().item())
        akhir = ukur(); net.load_state_dict(terbaik[1]); dini = ukur()
    return kurva, akhir, dini, terbaik[2]

hasil = {}
print("                   epoch 300       | early stopping")
print("                   NLL   akurasi   | epoch  NLL   akurasi")
for nama, kw in [("tanpa regularisasi", {}), ("weight decay 1,0", {"wd": 1.0}),
                 ("dropout 0,5", {"p_drop": 0.5})]:
    kurva, (n1, a1), (n2, a2), ep = latih(**kw); hasil[nama] = kurva
    print(f"{nama:18s} {n1:.2f}  {a1:.3f}     | {ep + 1:4d}   {n2:.2f}  {a2:.3f}")
np.savetxt("_kurva_reg.txt", np.c_[np.arange(1, 301), *hasil.values()], fmt="%.4f")
