"""Jalankan semua skrip di kode/ dan cocokkan keluarannya dengan keluaran di buku.

Pemakaian (dari folder akar repositori):
    python cek_keluaran.py            # semua bab
    python cek_keluaran.py 13 18      # hanya Bab 13 dan 18

Keluaran yang dicetak di buku tersimpan di kode/babNN/keluaran/<skrip>.txt.
Skrip disalin ke folder sementara agar berkas hasil tidak mengotori repositori.
Untuk setiap skrip dicetak OK (sama), XX (beda, dengan diff), atau !! (gagal).
Selisih kecil wajar bila versi pustaka berbeda, dan baris waktu eksekusi
("waktu ... s") selalu bergantung pada mesin.
"""
import sys, shutil, tempfile, subprocess, difflib, pathlib

AKAR = pathlib.Path(__file__).resolve().parent
# skrip yang memakai berkas hasil skrip lain di bab yang sama
SYARAT = {"bab05/laplace_logistik.py": "bab05/logistik_irls.py"}
pilih = {f"{int(a):02d}" for a in sys.argv[1:]}
tmp = pathlib.Path(tempfile.mkdtemp())
shutil.copytree(AKAR / "kode", tmp / "kode", ignore=shutil.ignore_patterns("__pycache__"))


def jalankan(f):
    return subprocess.run([sys.executable, f.name], cwd=f.parent,
                          capture_output=True, text=True, timeout=1800)


hasil = {"OK": 0, "XX": 0, "!!": 0}
for harap in sorted(tmp.glob("kode/bab*/keluaran/*.txt")):
    bab = harap.parent.parent
    if pilih and bab.name[3:] not in pilih:
        continue
    f = bab / (harap.stem + ".py"); nama = f"{bab.name}/{f.name}"
    if nama in SYARAT:
        jalankan(tmp / "kode" / SYARAT[nama])
    try:
        r = jalankan(f)
    except subprocess.TimeoutExpired:
        print(f"!! {nama}: timeout"); hasil["!!"] += 1; continue
    buku = [l.rstrip() for l in harap.read_text(encoding="utf-8").strip("\n").splitlines()]
    jalan = [l.rstrip() for l in r.stdout.strip("\n").splitlines()]
    if r.returncode != 0:
        print(f"!! {nama}: {r.stderr.strip()[-300:]}"); hasil["!!"] += 1
    elif buku == jalan:
        print(f"OK {nama}"); hasil["OK"] += 1
    else:
        print(f"XX {nama}"); hasil["XX"] += 1
        for l in difflib.unified_diff(buku, jalan, "buku", "jalan", lineterm="", n=0):
            print("   ", l)
    sys.stdout.flush()
print(f"\nringkasan: {hasil['OK']} cocok, {hasil['XX']} beda, {hasil['!!']} gagal")
shutil.rmtree(tmp, ignore_errors=True)
