# Fine-tuning IndoBERT untuk analisis sentimen ulasan (kerangka).
# Memerlukan: pip install transformers datasets, dan koneksi internet untuk
# mengunduh model (sekitar 500 MB). Skrip ini TIDAK dijalankan untuk buku ini.
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_ulasan import buat_data

nama = "indobenchmark/indobert-base-p1"                       # model fondasi  #<1>
tok = AutoTokenizer.from_pretrained(nama)
model = AutoModelForSequenceClassification.from_pretrained(nama, num_labels=2)  #<2>

teks, y = buat_data()                                         # ganti dengan data nyata
enc = tok(teks[:5000], padding=True, truncation=True, max_length=64, return_tensors="pt")
label = torch.tensor(y[:5000])
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)          # laju belajar kecil  #<3>
model.train()
for ep in range(2):
    for idx in torch.randperm(5000).split(32):
        keluar = model(input_ids=enc["input_ids"][idx],
                       attention_mask=enc["attention_mask"][idx], labels=label[idx])
        opt.zero_grad(); keluar.loss.backward(); opt.step()   # loss = cross-entropy
