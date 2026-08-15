"""M5: row-over-row correlation of Q changes vs previous gen's outcomes."""
import numpy as np

cols = list(range(7))
# live search root Q (opener persp), from M3/M3b, 48 runs each
liveQ = {
    6:  [-0.652, -0.423, -0.583, -0.120, -0.324, -0.376, -0.541],
    7:  [-0.468, -0.315, -0.384, -0.211, -0.151, -0.452, -0.559],
    8:  [-0.525,  0.084, -0.171, -0.294, -0.177, -0.174, -0.516],
    9:  [-0.481, -0.018, -0.401, -0.240, -0.270, -0.023, -0.191],
    10: [-0.434,  0.162, -0.465, -0.252, -0.357,  0.221, -0.294],
    11: [-0.279, -0.009, -0.627, -0.213, -0.416,  0.114, -0.192],
    12: [-0.070, -0.114, -0.515,  0.268, -0.182, -0.052,  0.175],
}
# per-gen mean ply-1 outcome target by opening col (opener persp), M2 (-v1|c)
outc = {
    6:  [-0.436, -0.283, -0.060, -0.249, -0.174, -0.120,  0.064],
    7:  [ 0.036, -0.332, -0.160, -0.202, -0.504, -0.063, -0.598],
    8:  [ 0.728,  0.029, -0.648, -0.090, -0.254, -0.057, -0.096],
    9:  [ 0.070, -0.159,  0.202, -0.141, -0.508,  0.033, -0.188],
    10: [ 0.424, -0.091, -0.255, -0.227,  0.295,  0.266, -0.862],
    11: [ 0.222, -0.125, -0.238,  0.034, -0.210, -0.174, -0.119],
}
# games per opening col, M2
cnt = {
    6:  [7, 9, 13, 424, 27, 15, 5],
    7:  [9, 20, 23, 135, 287, 15, 11],
    8:  [1, 209, 68, 77, 63, 77, 5],
    9:  [7, 190, 12, 96, 15, 169, 11],
    10: [4, 146, 7, 32, 7, 303, 1],
    11: [14, 70, 5, 38, 5, 359, 9],
}
# stored root visit dist, M2
vis = {
    6:  [0.010, 0.020, 0.029, 0.844, 0.054, 0.031, 0.012],
    7:  [0.017, 0.036, 0.048, 0.277, 0.586, 0.024, 0.013],
    8:  [0.008, 0.419, 0.138, 0.160, 0.141, 0.124, 0.010],
    9:  [0.009, 0.403, 0.023, 0.153, 0.040, 0.339, 0.032],
    10: [0.007, 0.307, 0.011, 0.072, 0.018, 0.577, 0.008],
    11: [0.014, 0.139, 0.010, 0.089, 0.020, 0.712, 0.015],
    12: [0.019, 0.024, 0.012, 0.820, 0.018, 0.027, 0.080],
}

def r(a, b, w=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if w is None:
        return np.corrcoef(a, b)[0, 1]
    w = np.asarray(w, float); w = w / w.sum()
    am, bm = (a*w).sum(), (b*w).sum()
    cov = (w*(a-am)*(b-bm)).sum()
    return cov / np.sqrt((w*(a-am)**2).sum() * (w*(b-bm)**2).sum())

print("hop      dQsearch vs outc(g-1)   dVisits vs outc(g-1)   [count-weighted r]")
for g in range(7, 13):
    p = g - 1
    dq = np.array(liveQ[g]) - np.array(liveQ[p])
    dv = np.array(vis[g]) - np.array(vis[p])
    o = np.array(outc[p]); w = np.array(cnt[p]) + 1
    print(f"{p}->{g}   r={r(dq,o):+.3f} (w {r(dq,o,w):+.3f})        "
          f"r={r(dv,o):+.3f} (w {r(dv,o,w):+.3f})")

print("\nhop table: dominant col at g-1, its outcome, its search-Q change, adopted col's Q change")
for g in range(7, 13):
    p = g - 1
    dom = int(np.argmax(vis[p]))
    newdom = int(np.argmax(vis[g]))
    dq_dom = liveQ[g][dom] - liveQ[p][dom]
    dq_new = liveQ[g][newdom] - liveQ[p][newdom]
    print(f"{p}->{g}: dom col{dom} (visits {vis[p][dom]:.2f}, outcome {outc[p][dom]:+.3f} "
          f"on {cnt[p][dom]} games) dQ={dq_dom:+.3f} -> visits {vis[g][dom]:.2f} | "
          f"adopted col{newdom}: dQ={dq_new:+.3f}, visits {vis[g][newdom]:.2f}")
