import numpy as np
import torch
from torch import nn

# Masalah invers (ilustrasi): t -> x mudah, x -> t bercabang.
rng = np.random.default_rng(0)
t = rng.uniform(0, 1, 1000)
x = t + 0.3 * np.sin(2 * np.pi * t) + rng.uniform(-0.1, 0.1, 1000)
X, T = torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(t[:, None], dtype=torch.float32)

K = 3
class MDN(nn.Module):                                         #<1>
    def __init__(self):
        super().__init__()
        self.badan = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.pi, self.mu, self.ls = nn.Linear(32, K), nn.Linear(32, K), nn.Linear(32, K)
    def forward(self, x):
        h = self.badan(x)
        return torch.log_softmax(self.pi(h), 1), self.mu(h), self.ls(h)  # log pi, mu, log sigma

def nll(lpi, mu, ls, t):                                       # -log-likelihood campuran  #<2>
    lp = lpi - ls - 0.5 * ((t - mu) / ls.exp()) ** 2 - 0.5 * np.log(2 * np.pi)
    return -torch.logsumexp(lp, 1).mean()

torch.manual_seed(0)
mdn, mlp = MDN(), nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
o1, o2 = torch.optim.Adam(mdn.parameters(), 3e-3), torch.optim.Adam(mlp.parameters(), 3e-3)
for it in range(4000):
    o1.zero_grad(); nll(*mdn(X), T).backward(); o1.step()
    o2.zero_grad(); ((mlp(X) - T) ** 2).mean().backward(); o2.step()   # kuadrat terkecil  #<3>
with torch.no_grad():
    print(f"-log-lik. per titik MDN: {nll(*mdn(X), T).item():.3f}")
    for x0 in [0.2, 0.5, 0.8]:                                 #<4>
        lpi, mu, ls = mdn(torch.tensor([[x0]]))
        kom = sorted(zip(lpi.exp()[0].tolist(), mu[0].tolist(), ls.exp()[0].tolist()), reverse=True)
        print(f"x = {x0}: kuadrat terkecil t = {mlp(torch.tensor([[x0]])).item():.2f}")
        print("   MDN: " + " + ".join(f"{p:.2f} N({m:.2f}, {s:.2f})" for p, m, s in kom if p > 0.05))
