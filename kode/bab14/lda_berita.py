import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from data_berita import buat_korpus, TOPIK

dok, theta, nama = buat_korpus()
latih, uji = dok[:1000], dok[1000:]
cv = CountVectorizer(token_pattern=r"[a-z]+")
Xl, Xu = cv.fit_transform(latih), cv.transform(uji)       # bag of words  #<1>
kata = cv.get_feature_names_out()

lda = LatentDirichletAllocation(6, doc_topic_prior=0.1, topic_word_prior=0.01,
                                learning_method="batch", random_state=0).fit(Xl)  #<2>
label = []
for k, b in enumerate(lda.components_):                   # kata teratas  #<3>
    atas = kata[np.argsort(b)[::-1][:6]]
    label.append(max(TOPIK, key=lambda t: len(set(atas) & set(TOPIK[t].split()))))
    print(f"topik {k} ({label[-1]:7s}): " + " ".join(atas))
d = 1000 + np.argmax(theta[1000:].max(1) < 0.7)         # dokumen uji campuran  #<4>
th = lda.transform(Xu[d - 1000])[0]
est = {label[k]: p for k, p in enumerate(th)}
print(f"dokumen #{d}  " + " ".join(f"{t:>7s}" for t in nama))
print("  taksiran    " + " ".join(f"{est.get(t, 0):7.2f}" for t in nama))
print("  sebenarnya  " + " ".join(f"{p:7.2f}" for p in theta[d]))
for K in [3, 6, 12]:                                      # perplexity data uji
    m = LatentDirichletAllocation(K, random_state=0).fit(Xl)
    print(f"K = {K:2d}: perplexity uji {m.perplexity(Xu):.1f}")
