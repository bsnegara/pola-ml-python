from scipy.stats import beta

# Prior dari angka nasional (ilustrasi): rerata 0,20, "setara" 20 balita
a0, b0 = 4, 16                                   # Beta(4, 16)  #<1>
print(f"prior : rerata {a0/(a0+b0):.3f}")

for nama, n, m in [("Desa A", 12, 5), ("Desa B", 120, 50)]:
    p_ml = m / n                                 # maximum likelihood  #<2>
    a, b = a0 + m, b0 + n - m                    # posterior konjugat  #<3>
    lo, hi = beta.ppf([0.025, 0.975], a, b)      # selang kredibel 95%
    print(f"{nama}: {m:2d}/{n:3d} | ML {p_ml:.3f} | "
          f"posterior {a/(a+b):.3f}, 95% [{lo:.3f}, {hi:.3f}]")
