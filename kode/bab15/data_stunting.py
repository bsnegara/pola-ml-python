"""Survei stunting sintetis di 30 kabupaten/kota (ilustrasi).

Ukuran sampel per kabupaten sangat bervariasi (12 sampai 400 balita), seperti
survei yang alokasi sampelnya tidak seimbang. Prevalensi sebenarnya diambil
dari distribusi logit-normal di sekitar 22%.
"""
import numpy as np
from scipy.special import expit

def buat_data(seed=7):
    rng = np.random.default_rng(seed)
    J = 30
    theta = expit(rng.normal(np.log(0.22 / 0.78), 0.35, J))   # prevalensi sebenarnya
    n = np.round(np.exp(rng.uniform(np.log(12), np.log(400), J))).astype(int)
    y = rng.binomial(n, theta)
    return y, n, theta
