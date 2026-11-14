import numpy as np

rng = np.random.default_rng(1)
sigma2 = 3.5**2                                  # varians sebenarnya: 12.25
for N in (2, 3, 5, 10, 50):
    X = rng.normal(85.0, 3.5, size=(200_000, N)) # 200 ribu "posyandu"  #<1>
    s2_ml = X.var(axis=1)                        # bagi N (ddof=0)  #<2>
    s2_tb = X.var(axis=1, ddof=1)                # bagi N-1 (tak bias)
    print(f"N={N:2d} | E[s2_ML] = {s2_ml.mean():5.2f} "
          f"(teori {(N-1)/N*sigma2:5.2f}) | E[s2_N-1] = {s2_tb.mean():5.2f}")
