"""Verify the agent's two OrthoGrad claims directly.

(1) On an output layer, OrthoGrad forbids the weight NORM from growing, so the
    head's output amplitude is stuck near its initialisation. value_1 is
    Dense(1, kernel_initializer='zeros'), i.e. it starts at norm 0 and must GROW
    its norm to express any magnitude at all.
(2) For a 1-element variable (value_1/bias), g_ortho is identically 0 and the
    rescale-to-||g|| step amplifies pure float cancellation noise.
"""
import numpy as np

rng = np.random.default_rng(0)


def orthogonalize(w, g):
    """Byte-for-byte the repo's math, in float32."""
    w = w.astype(np.float32).ravel()
    g = g.astype(np.float32).ravel()
    w_norm_sq = np.dot(w, w) + np.float32(1e-30)
    proj = np.dot(w, g) / w_norm_sq
    g_ortho = g - proj * w
    g_norm = np.linalg.norm(g)
    g_ortho_norm = np.linalg.norm(g_ortho) + np.float32(1e-30)
    return g_ortho * (g_norm / g_ortho_norm)


print("(1) CAN THE OUTPUT NORM GROW?  Dense(1) head, 64 inputs, lr 3e-4")
print("    target: the head must reach ||w|| ~ 0.2-0.8 to emit +-1 outputs\n")
for label, use_ortho in (("plain gradient", False), ("orthograd", True)):
    w = np.zeros(64, dtype=np.float32)          # kernel_initializer='zeros'
    lr = np.float32(3e-4)
    # a consistent teaching signal: the loss always wants w to move toward +d
    d = rng.normal(size=64).astype(np.float32)
    d /= np.linalg.norm(d)
    norms = []
    for step in range(1, 20001):
        g = -d + np.float32(0.3) * rng.normal(size=64).astype(np.float32)
        if use_ortho and np.dot(w, w) > 0:
            g = orthogonalize(w, g)
        w = w - lr * g
        if step in (100, 1000, 5000, 20000):
            norms.append((step, float(np.linalg.norm(w))))
    print(f"  {label:<16} " + "  ".join(f"step {s}: |w|={n:.4f}" for s, n in norms))

print("\n(2) THE 1-ELEMENT CASE (value_1/bias, shape (1,))")
n_zero = n_same = n_opp = 0
mags = []
for _ in range(200000):
    w = rng.normal(size=1).astype(np.float32)
    g = rng.normal(size=1).astype(np.float32)
    out = orthogonalize(w, g)
    if out[0] == 0.0:
        n_zero += 1
    elif np.sign(out[0]) == np.sign(g[0]):
        n_same += 1
        mags.append(abs(out[0]) / abs(g[0]))
    else:
        n_opp += 1
        mags.append(abs(out[0]) / abs(g[0]))
tot = 200000
print(f"  exactly zero : {n_zero / tot:.1%}")
print(f"  same sign    : {n_same / tot:.1%}")
print(f"  OPPOSITE sign: {n_opp / tot:.1%}   <- gradient ASCENT on the bias")
if mags:
    print(f"  |out|/|g| when nonzero: median {np.median(mags):.3f}")
