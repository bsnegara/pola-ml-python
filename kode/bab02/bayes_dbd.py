import numpy as np

# Angka ilustrasi (bukan data klinis): pasien demam di puskesmas saat musim hujan
prior = 0.05            # p(DBD): 5 dari 100 pasien demam  #<1>
sens = 0.90             # p(positif | DBD)
spes = 0.95             # p(negatif | bukan DBD)

# Teorema Bayes
p_pos = sens * prior + (1 - spes) * (1 - prior)        # aturan jumlah  #<2>
post = sens * prior / p_pos                             # aturan kali / Bayes
print(f"p(positif)         = {p_pos:.4f}")
print(f"p(DBD | positif)   = {post:.3f}")

# Periksa dengan simulasi: satu juta pasien  #<3>
rng = np.random.default_rng(0)
dbd = rng.random(1_000_000) < prior
positif = np.where(dbd, rng.random(dbd.size) < sens, rng.random(dbd.size) > spes)
print(f"simulasi: {positif.sum()} positif, {(dbd & positif).sum()} di antaranya DBD"
      f" -> {(dbd & positif).sum() / positif.sum():.3f}")
