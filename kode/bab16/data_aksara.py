"""Citra tulisan tangan sintetis 20 x 20 piksel untuk 10 kelas glif (ilustrasi).

Setiap glif dibangun dari goresan sederhana (lengkung, lingkaran kecil, garis
tegak dan mendatar) yang meniru gaya goresan aksara Jawa secara skematis.
Bentuknya TIDAK akurat sebagai aksara; untuk penelitian, gunakan himpunan data
tulisan tangan aksara Jawa yang sebenarnya.
"""
import numpy as np
from PIL import Image, ImageDraw

def _busur(cx, cy, r, a0, a1, n=12):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))

def _glif():
    G = []
    G.append([_busur(.3, .5, .18, 180, 360) + [(.48, .8)], [(.48, .5), (.7, .5)], _busur(.7, .5, .15, 180, 360)])
    G.append([[(.15, .7), (.15, .35)], _busur(.32, .35, .17, 180, 360), [(.49, .35), (.49, .7)], _busur(.66, .35, .17, 180, 360) + [(.83, .7)]])
    G.append([_busur(.5, .4, .25, 180, 360), [(.25, .4), (.25, .75), (.75, .75), (.75, .4)], _busur(.5, .55, .07, 0, 360)])
    G.append([[(.2, .3), (.8, .3)], [(.35, .3), (.35, .8)], [(.65, .3), (.65, .8)], _busur(.5, .8, .15, 0, 180)])
    G.append([_busur(.35, .5, .2, 90, 450), [(.55, .5), (.85, .5)], [(.85, .25), (.85, .8)]])
    G.append([_busur(.5, .35, .25, 180, 360), [(.25, .35), (.4, .8)], [(.75, .35), (.6, .8)], [(.4, .8), (.6, .8)]])
    G.append([[(.2, .25), (.2, .75)], [(.2, .5), (.8, .5)], [(.8, .25), (.8, .75)], _busur(.5, .3, .1, 0, 360)])
    G.append([_busur(.3, .4, .15, 0, 360), _busur(.7, .4, .15, 0, 360), [(.15, .75), (.85, .75)]])
    G.append([[(.15, .3), (.35, .75), (.5, .4), (.65, .75), (.85, .3)]])
    G.append([_busur(.5, .5, .3, 200, 520), [(.5, .2), (.5, .8)]])
    return G

GLIF = _glif()

def gambar(k, rng, ukuran=20):
    S = 64; im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
    a = np.radians(rng.normal(0, 12)); sk = rng.normal(1, 0.12, 2); sh = rng.normal(0, 0.2)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[sk[0], sh], [0, sk[1]]])
    geser = rng.normal(0, 0.06, 2)
    for gores in GLIF[k]:
        P = (np.array(gores) - 0.5 + rng.normal(0, 0.015, (len(gores), 2))) @ M.T + 0.5 + geser
        d.line([tuple(p) for p in P * S], fill=255, width=int(rng.integers(3, 6)), joint="curve")
    x = np.asarray(im.resize((ukuran, ukuran), Image.BILINEAR), float) / 255
    return np.clip(x + rng.normal(0, 0.1, x.shape), 0, 1)

def buat_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    X = np.stack([gambar(k, rng) for k in y]).reshape(n, -1)
    return X.astype(np.float32), y
