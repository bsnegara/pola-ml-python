"""Ulasan pendek layanan daring berbahasa Indonesia yang dibangkitkan dari pola
(ilustrasi). Polaritas dibalik oleh negasi ("tidak", "kurang", "nggak"), dan
pada kalimat "X tapi Y" yang menentukan adalah klausa kedua.
"""
import numpy as np

SUBJEK = ["drivernya", "pengirimannya", "aplikasinya", "barangnya", "pelayanannya",
          "harganya", "kurirnya", "makanannya", "penjualnya", "kemasannya"]
POS = ["bagus", "cepat", "ramah", "murah", "memuaskan", "mantap", "rapi", "enak"]
NEG = ["buruk", "lambat", "kasar", "mahal", "mengecewakan", "lemot", "berantakan", "hambar"]
KETERANGAN = ["", "", "sangat", "cukup", "agak", "benar-benar"]
NEGASI = ["tidak", "kurang", "nggak"]
PENGISI = ["", "", "", "kak", "sih", "ya", "deh", "pokoknya"]

def _klausa(rng):
    s = rng.choice(SUBJEK); pos = rng.random() < 0.5
    adj = rng.choice(POS if pos else NEG)
    neg = rng.random() < 0.35
    if neg:                                              # "kurang" hanya untuk kata positif
        ng = rng.choice(NEGASI if pos else ["tidak", "nggak"])
        kata = [s, ng, rng.choice(["", "", "begitu"] + ([] if ng == "kurang" else ["terlalu"])), adj]
    else:
        kata = [s, rng.choice(KETERANGAN), adj]
    return [w for w in kata if w], pos != neg            # negasi membalik polaritas

def buat_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    teks, y = [], []
    for _ in range(n):
        k1, p1 = _klausa(rng)
        if rng.random() < 0.4:                           # "X tapi Y"
            k2, p2 = _klausa(rng)
            kata, p = k1 + ["tapi"] + k2, p2
        else:
            kata, p = k1, p1
        isi = rng.choice(PENGISI)
        teks.append(" ".join(kata + ([isi] if isi else []))); y.append(int(p))
    return teks, np.array(y)
