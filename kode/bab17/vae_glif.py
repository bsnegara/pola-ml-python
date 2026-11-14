import sys, pathlib
import torch
from torch import nn
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bab16"))
from data_aksara import buat_data                            # glif dari Bab 16

X, y = buat_data()
X = torch.tensor(X); Xl, Xu = X[:5000], X[5000:]
Z = 2                                                         # dimensi laten

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(400, 256), nn.ReLU(), nn.Linear(256, 2 * Z))  #<1>
        self.dec = nn.Sequential(nn.Linear(Z, 256), nn.ReLU(), nn.Linear(256, 400))
    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, 1)                 # q(z | x) Gaussian
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) # reparameterisasi  #<2>
        return self.dec(z), mu, logvar

def elbo(net, x):                                             #<3>
    logit, mu, logvar = net(x)
    rek = -nn.functional.binary_cross_entropy_with_logits(logit, x, reduction="none").sum(1)
    kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)  # KL(q || N(0, I))
    return rek.mean(), kl.mean()

torch.manual_seed(0)
net = VAE(); opt = torch.optim.Adam(net.parameters(), 1e-3)
for ep in range(30):
    for idx in torch.randperm(5000, generator=torch.Generator().manual_seed(ep)).split(64):
        rek, kl = elbo(net, Xl[idx]); opt.zero_grad(); (kl - rek).backward(); opt.step()
    if ep % 10 == 9:
        with torch.no_grad():
            rek, kl = elbo(net, Xu)
        print(f"epoch {ep + 1}: ELBO uji {(rek - kl).item():7.1f} = rekonstruksi {rek.item():7.1f}"
              f" - KL {kl.item():4.1f}")
from sklearn.neighbors import KNeighborsClassifier           # seberapa informatif z?  #<4>
with torch.no_grad():
    zl, zu = net.enc(Xl).chunk(2, 1)[0], net.enc(Xu).chunk(2, 1)[0]
knn = KNeighborsClassifier(10).fit(zl, y[:5000])
from sklearn.decomposition import PCA
pca = PCA(2).fit(Xl.numpy())
knn_p = KNeighborsClassifier(10).fit(pca.transform(Xl.numpy()), y[:5000])
print(f"akurasi 10-NN di ruang 2D: VAE {knn.score(zu, y[5000:]):.3f},"
      f" PCA {knn_p.score(pca.transform(Xu.numpy()), y[5000:]):.3f}")
