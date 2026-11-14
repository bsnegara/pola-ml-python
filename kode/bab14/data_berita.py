"""Korpus berita daerah sintetis yang dibangkitkan dari model LDA (ilustrasi).

Enam topik dengan kosakata khasnya, ditambah kata umum yang muncul di semua
topik. Setiap dokumen memiliki campuran topiknya sendiri.
"""
import numpy as np

TOPIK = {
    "sawit":    "sawit kebun petani tbs harga pabrik plasma replanting cpo lahan "
                "produksi koperasi bibit pupuk ekspor",
    "banjir":   "banjir hujan sungai warga pengungsi bpbd tanggul evakuasi debit "
                "longsor genangan bantuan perahu rendam siaga",
    "pilkada":  "pilkada calon kpu suara pemilih kampanye debat bawaslu partai "
                "koalisi survei tps paslon dukungan pencoblosan",
    "wisata":   "wisata pantai wisatawan hotel festival budaya kuliner danau homestay "
                "tiket libur pengunjung desa pesona promosi",
    "bola":     "pemain gol klub pelatih liga laga stadion suporter skor kompetisi "
                "babak tim juara kiper menang",
    "dbd":      "dbd nyamuk fogging puskesmas pasien demam kasus jentik dinkes rsud "
                "trombosit abate pencegahan kesehatan",
}
UMUM = "kata pemerintah daerah kabupaten provinsi tahun hari masyarakat kota bupati".split()

def buat_korpus(n_dok=1200, panjang=80, alpha=0.15, seed=0):
    rng = np.random.default_rng(seed)
    kosakata = sorted({w for v in TOPIK.values() for w in v.split()} | set(UMUM))
    iw = {w: i for i, w in enumerate(kosakata)}
    phi = np.full((len(TOPIK), len(kosakata)), 1e-3)     # distribusi kata per topik
    for k, v in enumerate(TOPIK.values()):
        kata = v.split()
        phi[k, [iw[w] for w in kata]] += rng.dirichlet(np.ones(len(kata)) * 2) * 0.75
        phi[k, [iw[w] for w in UMUM]] += 0.25 / len(UMUM)
    phi /= phi.sum(1, keepdims=True)
    theta = rng.dirichlet(np.ones(len(TOPIK)) * alpha, n_dok)  # campuran per dokumen
    dok = []
    for d in range(n_dok):
        z = rng.choice(len(TOPIK), panjang, p=theta[d])
        dok.append(" ".join(kosakata[rng.choice(len(kosakata), p=phi[k])] for k in z))
    return dok, theta, list(TOPIK)
