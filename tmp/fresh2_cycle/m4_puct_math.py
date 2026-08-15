"""M4: magnitude check. Fixed-Q PUCT bandit at the root, 375 sims, c_puct 1.5.

Exact selection rule replicated from MCTS.py: unvisited children Q=0,
U = P * sqrt(N_parent) / (n+1) * (c_init + ln((N_parent + c_base + 1)/c_base)),
first 7 sims expand every child once (Self_Play's fully_visited loop).

Question: how big a Q gap does it take for a P=0.105 child (col5, gen-11 prior)
to out-visit the P=0.477 center? Sweep dQ. Also run with the ACTUAL measured
gen-11 live root Qs.
"""
import numpy as np

CB = 19_652.0
CI = 1.5
SIMS = 375

def bandit(P, Q, sims=SIMS):
    n = np.zeros(7); N = 0
    # replicate 'fully visited first' phase
    order = np.argsort(P)[::-1]
    for c in order:
        n[c] += 1; N += 1
    for _ in range(sims - 7):
        q = Q.copy()
        u = P * (N ** 0.5) / (n + 1) * (CI + np.log((N + CB + 1) / CB))
        best = np.argmax(q + u)
        n[best] += 1; N += 1
    return n / N

prior11 = np.array([0.048, 0.110, 0.098, 0.477, 0.136, 0.105, 0.026])

print("A) dQ sweep: Q(col5) = Q(others) + dQ, all others equal at -0.1")
print(f"{'dQ':>6} {'share(col3)':>12} {'share(col5)':>12}")
for dq in (0.0, 0.02, 0.05, 0.10, 0.20, 0.33):
    Q = np.full(7, -0.1); Q[5] += dq
    s = bandit(prior11, Q)
    print(f"{dq:>6.2f} {s[3]:>12.3f} {s[5]:>12.3f}")

print("\nB) actual gen-11 measured live root Q (from M3), gen-11 prior:")
Q11 = np.array([-0.279, -0.009, -0.627, -0.213, -0.416, 0.114, -0.192])
s = bandit(prior11, Q11)
print("            " + "".join(f"  col{c}  " for c in range(7)))
print("bandit dist " + "".join(f" {s[c]:6.3f} " for c in range(7)))
print("live search " + "".join(f" {x:6.3f} " for x in
      [0.012, 0.137, 0.011, 0.091, 0.019, 0.718, 0.012]))

print("\nC) U-term magnitude at N=375 for col3 (P=0.477) vs col5 (P=0.105):")
N = 375.0
mult = CI + np.log((N + CB + 1) / CB)
for name, p, nvis in (("col3", 0.477, 34), ("col5", 0.105, 269)):
    u = p * (N ** 0.5) / (nvis + 1) * mult
    print(f"  {name}: P={p}, n={nvis} -> U = {u:.4f}")
print(f"  (multiplier c_init+ln(...) = {mult:.4f}, sqrt(N) = {N**0.5:.2f})")
print("  => at the observed visit split, col3's U bonus is ~0.40 vs col5's ~0.01;")
print("     a Q gap must exceed that residual ~0.39 U-difference to HOLD the split,")
print("     i.e. the 0.33 measured Q gap plus deeper-in-tree reinforcement suffices.")

print("\nD) typical net value noise vs required gap: mirror-twin fit errors")
print("   (targets exactly equal, so |Q(c)-Q(6-c)| is pure fit error):")
print("   gen 8: 0.21-0.56, gen 10: 0.22-0.45, gen 11: 0.07-0.48")
print("   -> fit error alone reaches the 0.05-0.20 range that section A shows")
print("      is enough to redirect the majority of root visits.")
