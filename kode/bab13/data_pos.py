"""Korpus kecil kalimat bahasa Indonesia berlabel kelas kata (sintetis).

Kalimat dibangkitkan dari pola kelas kata dan leksikon sederhana. Beberapa kata
sengaja taksa (ambigu), misalnya "tahu" (NOUN: makanan, VERB: mengetahui),
"bisa" (MD: dapat, NOUN: racun), "jalan", "masak", dan "baru".
"""
import numpy as np

TAG = ["PRON", "NOUN", "VERB", "ADJ", "ADP", "ADV", "MD", "DET"]
LEKSIKON = {
    "PRON": ["saya", "dia", "kami", "mereka", "kita", "aku", "kamu", "beliau"],
    "NOUN": ["nasi", "tahu", "tempe", "pasar", "sekolah", "rumah", "jalan", "buku",
             "kopi", "sawah", "bisa", "ular", "sore", "pagi", "kantor", "kereta"],
    "VERB": ["makan", "masak", "membeli", "pergi", "membaca", "tahu", "jalan",
             "minum", "menjual", "datang", "pulang", "melihat"],
    "ADJ":  ["enak", "panas", "baru", "besar", "murah", "masak", "ramai", "jauh"],
    "ADP":  ["di", "ke", "dari", "untuk", "dengan"],
    "ADV":  ["baru", "sudah", "sedang", "sering", "tidak", "juga"],
    "MD":   ["bisa", "akan", "harus", "mau", "boleh"],
    "DET":  ["itu", "ini"],
}
POLA = [
    "PRON VERB NOUN", "PRON VERB NOUN ADJ", "PRON ADV VERB NOUN",
    "PRON MD VERB NOUN ADP NOUN", "NOUN DET ADJ", "NOUN DET ADV ADJ",
    "PRON VERB ADP NOUN", "PRON ADV VERB ADP NOUN NOUN", "NOUN NOUN DET ADJ",
    "PRON MD VERB ADP NOUN", "PRON VERB NOUN DET", "NOUN ADJ DET ADV ADJ",
    "PRON ADV MD VERB NOUN", "PRON VERB NOUN ADP NOUN DET",
]

def buat_korpus(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    p_pola = rng.dirichlet(np.ones(len(POLA)) * 2)
    korpus = []
    for _ in range(n):
        tags = POLA[rng.choice(len(POLA), p=p_pola)].split()
        kata = [LEKSIKON[t][min(int(rng.exponential(len(LEKSIKON[t]) / 3)),
                                len(LEKSIKON[t]) - 1)] for t in tags]  # Zipf kasar
        korpus.append(list(zip(kata, tags)))
    return korpus
