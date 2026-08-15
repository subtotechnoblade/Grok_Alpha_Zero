"""M12: unit-test the two formulas the whole report rests on, on synthetic data
where mu and sigma^2 are KNOWN.

  (1) keras reported loss with weights normalised to mean 1
          = (1/N_raw) * sum_over_unique n_j (pred_j - ybar_j)^2
  (2) an oracle predicting the true mu_j scores
          (1/N_raw) * sum_over_unique mult_j * sigma_j^2
"""
import numpy as np

rng = np.random.RandomState(7)
G = 4000
# occurrence counts with a real-looking heavy tail
n = np.maximum(1, rng.zipf(1.6, G)).astype(int)
n[n > 400] = 400
N = n.sum()
mu = rng.uniform(-0.9, 0.9, G)
# z in {-1,0,1} with mean mu and P(0)=p0 -> sigma^2 = (1-p0) - mu^2
p0 = rng.uniform(0.0, 0.3, G)
pp = (1 - p0 + mu) / 2
pm = (1 - p0 - mu) / 2
ok = (pp >= 0) & (pm >= 0)
mu, p0, pp, pm, n = mu[ok], p0[ok], pp[ok], pm[ok], n[ok]
G = len(mu); N = n.sum()
sigma2 = (1 - p0) - mu ** 2

TRIALS = 400
tot = 0.0
for _ in range(TRIALS):
    ybar = np.empty(G)
    for j in range(G):
        u = rng.rand(n[j])
        z = np.where(u < pp[j], 1.0, np.where(u < pp[j] + pm[j], -1.0, 0.0))
        ybar[j] = z.mean()
    # formula (1) with pred = mu (the oracle)
    tot += (n * (mu - ybar) ** 2).sum() / N
emp = tot / TRIALS
pred = (sigma2).sum() / N          # mult_j == 1 here
print(f"[oracle dedup loss]  empirical {emp:.6f}   formula {pred:.6f}   "
      f"ratio {emp/pred:.4f}   (G={G} groups, N={N} raw samples)")

# formula (1) vs an explicit mean-1-normalised weighted average, arbitrary predictor
p = rng.uniform(-1, 1, G)
w = n * (G / n.sum())
lhs = (w * (p - ybar) ** 2).sum() / G          # keras: sum(w*l)/n_dedup_samples
rhs = (n * (p - ybar) ** 2).sum() / N
print(f"[keras reduction]    sum(w*l)/U = {lhs:.8f}   (1/N)sum n_j l_j = {rhs:.8f}   "
      f"equal: {np.isclose(lhs, rhs)}   mean(w)={w.mean():.6f}")

# mirror multiplicity: duplicate every draw twice (mult=2) -> loss must DOUBLE per group
n2 = n * 2
lhs2 = (n2 * (mu - ybar) ** 2).sum() / n2.sum()
print(f"[mult=2 invariance]  duplicated-copy loss {lhs2:.6f} vs single {(n*(mu-ybar)**2).sum()/N:.6f} "
      f"(ratio {lhs2/((n*(mu-ybar)**2).sum()/N):.4f}; ybar unchanged so the objective is unchanged, "
      f"but the FLOOR formula must use n_eff, not n)")
