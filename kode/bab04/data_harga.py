"""Data sintetis harga cabai merah mingguan (ilustrasi, bukan data resmi)."""
import numpy as np

MINGGU_LEBARAN = [18, 16+52, 15+104, 13+156]      # Idulfitri 2022-2025

def pola_harga(t):
    """Harga 'sebenarnya' (ribu Rp/kg) pada minggu ke-t sejak Januari 2022."""
    tren = 40 + 0.03 * t                           # naik pelan
    musim = 9 * np.cos(2*np.pi*t/52) + 3 * np.sin(4*np.pi*t/52)  # puncak musim hujan
    lebaran = sum(8 * np.exp(-0.5 * ((t - L) / 1.5)**2) for L in MINGGU_LEBARAN)
    return tren + musim + lebaran

def buat_data(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(208)                             # 4 tahun, mingguan
    y = pola_harga(t) + rng.normal(0, 3.0, t.size) # derau pasar, sd Rp3.000
    return t, y

def fitur(t, lebaran=True, tren=True, K=2):
    """Fungsi basis: konstanta, tren, Fourier tahunan, benjolan Lebaran."""
    kol = [np.ones_like(t, dtype=float)]
    if tren:
        kol.append(t / 52)
    for k in range(1, K + 1):
        kol += [np.cos(2*np.pi*k*t/52), np.sin(2*np.pi*k*t/52)]
    if lebaran:
        kol.append(sum(np.exp(-0.5*((t - L)/1.5)**2) for L in MINGGU_LEBARAN))
    return np.column_stack(kol)
