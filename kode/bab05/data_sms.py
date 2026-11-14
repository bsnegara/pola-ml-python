"""Korpus SMS sintetis berbahasa Indonesia (ilustrasi, bukan pesan asli).

Pesan dibangkitkan dari templat yang meniru pola umum penipuan (hadiah,
"mama minta pulsa", pinjol, kurir/APK, perubahan tarif bank) dan pesan
sehari-hari. Semua nomor, tautan, dan nama adalah rekaan.
"""
import numpy as np

PENIPUAN = {
    "hadiah": ["SELAMAT! No anda terpilih mendapat hadiah {hadiah} dari {usaha}. Info klik {url} PIN: {pin}",
               "Selamat anda pemenang undian {usaha} ke-{pin}, hadiah {hadiah}. Hub {hp}",
               "PEMENANG {hadiah}!! Segera klaim di {url} sebelum hangus"],
    "keluarga": ["Ma ini nomor baru papa, tolong kirim pulsa {nominal} ke nomor ini dulu ya penting",
                 "Bu ini aku, hp ku rusak. tolong transfer {rp} ke rek {bank} {norek} a/n {nama}",
                 "Pak, saya {nama} temen anak bapak, dia kecelakaan, butuh {rp} sekarang ke {norek}",
                 "TF {rp} ke {norek} {bank} skrg, konfirm {hp}",
                 "Kak ini aku, bisa pinjam {rp} dulu? nanti diganti hari {hari}"],
    "pinjol": ["Pinjaman dana cepat cair {rp} tanpa jaminan, bunga 0%. Daftar via WA {hp}",
               "BUTUH DANA? cair 5 menit {rp}, cukup KTP. klik {url}",
               "Limit pinjaman anda {rp} sudah aktif, ambil sekarang di {url}"],
    "kurir": ["Paket anda tertahan. Lihat foto resi di aplikasi berikut: {apk}",
              "Kurir {kurir}: paket tidak dapat dikirim, konfirmasi alamat di {url}",
              "Ini foto paketnya kak, tolong dicek {apk}"],
    "bank": ["Nasabah {bank} Yth, tarif transfer berubah jadi Rp150.000/bln. Jika tidak setuju isi {url}",
             "Akun {bank} anda DIBLOKIR. Verifikasi data di {url} dalam 1x24 jam",
             "Poin {bank} anda {pin} akan hangus, tukarkan di {url}"],
}
NORMAL = ["{sapa}, nanti jam {jam} jadi ketemu di {tempat}?",
          "Bu, besok rapat prodi jam {jam} di ruang {ruang}",
          "Mas, uang kos bulan ini sudah saya transfer ya {rp}",
          "Pesanan kamu sudah sampai, ditaruh di depan pintu",
          "Kode OTP anda {pin}. JANGAN berikan kode ini kepada siapapun",
          "Paket {kurir} dengan resi {resi} sedang diantar kurir",
          "Jangan lupa beli {barang} pas pulang ya",
          "Assalamualaikum, kajian ba'da maghrib di masjid {masjid} malam ini",
          "Tagihan listrik bulan ini {rp}, sudah dibayar belum?",
          "Selamat ulang tahun {nama}! semoga sehat selalu",
          "{sapa} tugas kelompok dikumpul hari {hari} ya, jangan telat",
          "Ok siap, nanti saya kabari lagi",
          "Pak, mohon izin hari ini tidak masuk karena sakit",
          "Transaksi {bank} sebesar {rp} berhasil pada {jam}",
          "Mak, aku pulang hari {hari} naik travel jam {jam}",
          "Promo {usaha}: diskon 50% hari ini, cek aplikasinya",
          "Pinjam dulu {rp} ya, nanti aku ganti hari {hari}",
          "No rek aku {bank} {norek} a/n {nama}, makasih ya"]

SELIPAN = ["ya", "deh", "kak", "sekarang", "info", "segera", "makasih", "penting",
           "cek", "tolong", "dulu", "bisa", "nanti", "hari", "ini", "aja", "gan", "bu"]

ISI = dict(hadiah=["mobil Avanza", "uang tunai 100jt", "motor NMAX", "emas 10gr"],
           usaha=["PT Sinar Jaya", "Shopee", "Bank Rakyat", "Telkomsel"],
           url=["bit.ly/hdh-{n}", "klaim-{n}.xyz", "s.id/promo{n}", "cek-resi{n}.top"],
           apk=["resi_paket.apk", "foto-paket{n}.apk", "lihat_resi.apk"],
           bank=["BRI", "BNI", "Mandiri", "BCA", "BSI"], kurir=["JNE", "J&T", "SiCepat", "POS"],
           nama=["Rina", "Andi", "Siti", "Budi", "Dewi", "Fajar"],
           sapa=["Bro", "Kak", "Dek", "Mbak", "Bang"], tempat=["kampus", "kantin", "masjid", "mall"],
           ruang=["301", "rapat", "dosen"], barang=["telur", "beras", "gas", "galon"],
           masjid=["Al-Ikhlas", "Raya", "Nurul Iman"], hari=["Senin", "Rabu", "Jumat", "Sabtu"])

def _isi(templat, rng):
    nilai = {k: rng.choice(v).replace("{n}", str(rng.integers(10, 999))) for k, v in ISI.items()}
    nilai.update(pin=str(rng.integers(1000, 99999)), resi=f"JP{rng.integers(10**9, 10**10)}",
                 hp=f"08{rng.integers(10**9, 10**10)}", norek=str(rng.integers(10**9, 10**10)),
                 jam=f"{rng.integers(7, 22)}.{rng.choice(['00', '30'])}",
                 rp="Rp" + f"{int(rng.integers(1, 50)) * 50_000:,}".replace(",", "."), nominal=f"{rng.choice([20, 50, 100])}rb")
    kata = templat.format(**nilai).split()
    if len(kata) > 4 and rng.random() < 0.5:           # pesan nyata tidak rapi:
        del kata[rng.integers(len(kata))]                # kata hilang,
    for _ in range(rng.poisson(1.0)):                    # kata tambahan acak
        kata.insert(rng.integers(len(kata) + 1), rng.choice(SELIPAN))
    return " ".join(kata)

def buat_korpus(n_normal=1500, n_penipuan=500, seed=0):
    """Kembalikan (teks, label 0/1, jenis penipuan atau 'normal')."""
    rng = np.random.default_rng(seed)
    teks, label, jenis = [], [], []
    for _ in range(n_normal):
        teks.append(_isi(rng.choice(NORMAL), rng)); label.append(0); jenis.append("normal")
    nama_jenis = list(PENIPUAN)
    for _ in range(n_penipuan):
        j = rng.choice(nama_jenis)
        teks.append(_isi(rng.choice(PENIPUAN[j]), rng)); label.append(1); jenis.append(j)
    balik = rng.random(len(label)) < 0.02               # 2% label keliru (anotator)
    label = [1 - l if b else l for l, b in zip(label, balik)]
    urut = rng.permutation(len(teks))
    return ([teks[i] for i in urut], np.array(label)[urut], np.array(jenis)[urut])

def _fitur_dua_tidak_dipakai(teks):
    """Dua fitur sederhana: proporsi angka dan proporsi huruf kapital."""
    angka = np.array([sum(c.isdigit() for c in s) / len(s) for s in teks])
    kapital = np.array([sum(c.isupper() for c in s) / len(s) for s in teks])
    return np.column_stack([angka, kapital])

def normalisasi(s):
    """Huruf kecil, setiap deret angka diganti '0' agar nomor tidak jadi kata unik."""
    import re
    return re.sub(r"\d+", "0", s.lower())

def kantong_kata(teks, latih):
    """Matriks bag of words biner; kosakata hanya dari data latih (min. 3 pesan)."""
    from sklearn.feature_extraction.text import CountVectorizer
    vek = CountVectorizer(preprocessor=normalisasi, token_pattern=r"[a-z0-9&]+",
                          min_df=3, binary=True)
    vek.fit([s for s, l in zip(teks, latih) if l])
    return vek.transform(teks).toarray().astype(float), vek.get_feature_names_out()
