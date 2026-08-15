# Grok_Zero audit + optimizer benchmark — findings

Everything below is measured, not inferred. Scripts are in `tmp/`, raw numbers in
`tmp/results_*.json`.

---

## 1. The question you asked last: OrthoGrad, yes or no?

**Answer: no. Turn it off.**

`tmp/exp_j_final.py` — 6 arms x 3 seeds x 6 epochs on CIFAR-10, paired design (a given
seed fixes weight init, shuffle order and the augmentation stream, so every arm sees
identical data and `ortho − baseline` is a within-seed difference).

```
arm                          test acc (last 2 ep)   train acc   |W| growth
adam                            0.7101 +/- 0.0291      0.8087        1.98x
adam + ortho(grad)              0.6928 +/- 0.0300      0.7999        1.89x
adam + ortho(update)            0.7212 +/- 0.0412      0.8124        1.98x
muon                            0.6918 +/- 0.0062      0.8315        1.51x
muon + ortho(grad)              0.6249 +/- 0.0572      0.8105        1.54x
muon + ortho(update)            0.6686 +/- 0.0315      0.8334        1.50x

PAIRED differences (ortho minus its own baseline)
comparison                  seed 7   seed 13   seed 29      mean       sd
adam + ortho(grad)         -0.0218   -0.0536   +0.0235   -0.0173   0.0316
adam + ortho(update)       +0.0777   -0.0023   -0.0420   +0.0111   0.0498
muon + ortho(grad)         -0.0584   -0.1341   -0.0082   -0.0669   0.0517
muon + ortho(update)       +0.0211   -0.0229   -0.0677   -0.0232   0.0363
```

No variant clears its own noise. But note `muon + ortho(grad)` — your current repo
config — is **negative on all three seeds**, and negative again in the independent
5-epoch sweep (0.7559 -> 0.6943). Four observations, four negative. That is the only
comparison with a consistent sign, and it points against the thing you are running.

The earlier +6.5 point result for `adam + ortho(update)` did **not** replicate. It was
cross-script and sat inside the noise band.

### Why it cannot work here (this part is solid, unlike the accuracy numbers)

`|W| growth` is the most reliable column in the table — across three seeds `adam` gives
1.97 / 1.98 / 1.99. And OrthoGrad barely moves it: 1.98 -> 1.89 (gradient projection),
1.98 -> 1.98 (update projection). It is not doing the job it exists to do.

Measured `cos(gradient, W)` on the trained model:

```
stem/kernel                       (3, 3, 3, 32)     cos +0.00000   (random ~0.0340)
res_net__block/conv2d/kernel      (3, 3, 32, 32)    cos +0.00000   (random ~0.0104)
res_net__block_1/conv2d_3/kernel  (3, 3, 32, 32)    cos +0.00001
res_net__block_4/conv2d_12/kernel (3, 3, 64, 128)   cos -0.00001
```

Exactly zero for every conv that feeds a BatchNorm. BN makes the loss scale-invariant in
that kernel, so the gradient is already perpendicular to `W` and OrthoGrad projects out a
component that is mathematically zero. It is a no-op by construction.

Moving the projection to the update (`tmp/post_ortho.py`) does not rescue it either, and
the reason is the part I originally had wrong: **weight norm growth under BN is not
caused by a radial gradient.** If every step is perfectly perpendicular to `W` with length
d, then `||W||` still grows as `sqrt(||W0||^2 + N*d^2)` — perpendicular vectors add in
quadrature. Over ~980 steps that alone produces the observed 1.98x. No projection can
stop it; only weight decay or an explicit renormalisation of `W` can.

Corollary: update projection and weight decay are **mutually exclusive**. Decoupled decay
is a purely radial update, so projecting orthogonal to `W` deletes it exactly.

**Action:** set `use_orthograd: False` in `Connect4/Connect4.py`. OrthoGrad's premise
holds in un-normalised networks — on a BN-free MLP it held `||W||` at 1.02x against bare
Muon's 1.95x — but your net is BatchNorm throughout.

---

## 2. What the Muon reshape bug actually cost

Keras conv kernels are `(kh, kw, in, out)`; Keller's reference impl is PyTorch, which is
`(out, in, kh, kw)`. `reshape(G, (G.shape[0], -1))` therefore produced a `(3, N)` matrix —
kernel *height* as the leading dim.

**Mechanism** (`tmp/exp_a_mechanism.py`, real gradients from the real Connect4 model):

| | buggy | fixed |
|---|---|---|
| NS matrix for a `(3,3,128,128)` kernel | `(3, 49152)` | `(128, 1152)` |
| `cos(update, gradient)` | 0.97 – 0.99 | 0.66 – 0.79 |
| per-output-channel update energy | min 0.46, max 1.96 | min 0.88, max 1.14 |
| `‖update‖` on trunk convs | 0.0036 | 0.0211 (5.9x larger) |
| `‖update‖` on head convs | 0.0151 | 0.0048 (3.1x smaller) |

Three singular values to equalise across 147k numbers is a near-vacuous constraint, so the
"orthogonalised" update stayed 97–99% aligned with the raw momentum gradient. **The buggy
Muon was normalised SGD wearing a Muon costume.** The trunk-to-head step ratio was also off
by ~18x.

**Cost, measured two ways:**

- **Connect 4 self-play data**: none detectable. Tuned against tuned, buggy 2.6051 +/- 0.0332
  vs fixed 2.6252 +/- 0.0298 val loss (3 seeds). The bug behaved as a ~3x learning rate
  mis-scaling and nothing more.
- **CIFAR-10**: catastrophic. 0.5003 vs 0.7559 test accuracy, fully bracketed over
  1e-3 … 1e-1 with an interior optimum. **25 accuracy points.**

The contrast is the lesson: self-play targets are soft, non-stationary and regenerated each
generation by a search that is itself changing. That benchmark has almost no power to
resolve an optimizer difference. Do optimizer work on a fixed supervised task.

---

## 3. Optimizer benchmark, CIFAR-10, 700k-param ResNet using your `ResNet_Block`

Best per arm after a full lr sweep (`tmp/exp_f_cifar.py`, 5 epochs, 1 seed — see the noise
caveat in §5):

```
adamw (control)        0.7728  @ 3e-3
muon (fixed)           0.7559  @ 1e-1
grokfast(muon)         0.7198  @ 3e-2
orthograd(muon)        0.6943  @ 1e-1
muon (buggy reshape)   0.5003  @ 1e-2
```

- **Grokfast is worse than plain Muon**, as you predicted. At 1e-1 it hits train 0.824 /
  test 0.617 — the widest generalisation gap of any arm.
- **Muon does not beat AdamW at this scale.** Muon's published wins are on large
  transformer matrices; a 700k-param net with 32/64/128-channel convs is not that.
- **Muon's optimal lr is 1e-1, 33x higher than AdamW's 3e-3.** Its update has fixed norm
  rather than adaptive scale.
- **Speed:** roughly a wash, 18–26 s/epoch for everything. Accuracy per second favours
  AdamW at short budgets, but Muon is still accelerating at epoch 5 (+0.166 in its last
  epoch vs AdamW's +0.034), so a short window flatters AdamW.
- **bfloat16 Newton-Schulz costs nothing on Turing** — 18.5 vs 18.3 s/epoch against a
  float32 variant, identical accuracy. I expected this to be a problem on a 2060. It isn't.

---

## 4. Connect 4 bugs found and fixed

| # | file | bug |
|---|---|---|
| 1 | `Connect4.py:452` | `np.fliplr` flipped **height**, not columns. Board is `(T,6,7,4)` so axis 1 is rows; policy is `(T,7)` so axis 1 is columns. **Half of every training set was gravity-violating boards paired with a mirrored policy.** |
| 2 | `Connect4.py:346` | history loop wrote `board_state[i-1]` for `i = -1,-2,-3` -> indices 2,1,**0**, overwriting the side-to-move plane. From ply 4 the net had no turn signal, and plane 0 changed meaning mid-game. |
| 3 | `Connect4.py:69` | `exclude_layers` named layers that did not exist, so both output heads went through Muon. |
| 4 | `Build_Model.py` | two layers named `policy` -> model would not build. |
| 5 | `MCTS.py:481` | Dirichlet noise applied at **every** node, not just the root. 25% noise at every interior node destroys deep search. Now root-only, re-drawn on re-root in `_set_root`. |
| 6 | `MCTS.py:373` | `len(terminal_mask == 1)` returns the array length, not the win count. Leaked into root visit counts = the policy target. |
| 7 | `MCTS.py:344` | won root back-propagated `value=1` for **drawing** children too. |
| 8 | `MCTS.py:240` | gen-0 leaf value was `uniform(-1,1)` — pure noise backed up the tree. Now 0.0. |
| 9 | `Dataloader.py` | train/test split ran over pooled positions **after** augmentation, so a board and its own mirror landed on opposite sides. `val_loss` tracked train loss, making `save_best_only` a no-op. Now splits by game id. |
| 10 | `muon.py` | conv reshape (§2) + `d_in`/`d_out` read off the wrong axes. |
| 11 | `ortho_grad.py`, `grok_fast.py` | wrapped optimizer's step counter never ticked, so Muon's Adam bias correction was frozen at `t=1` forever (`1-b1^t = 0.1`). |
| 12 | `muon.py` | `self.weight_decay` is also Keras' own attribute, so decay applied **twice** when unwrapped. Renamed to `muon_weight_decay`. |
| 13 | all 3 optimizers | `get_config()` built the dict and never returned it. |

Config changes: exploration plies 15 -> 6 (was ~47% of an average game at tau=1),
`c_puct_init` 3.5 -> 1.5, added an lr schedule with base 3e-3 (was a flat 1e-2 for 100
generations), `train_epochs` 15 -> 8, `q_weight` exposed at 0.5, `.gitignore` added
(9.5 GB of run data was untracked and unignored).

**Connect 4's learning rate was ~3x too high regardless of the bug** — buggy@1e-2 scored
2.919 val loss vs buggy@3e-3's 2.605, ~8 sigma. That was costing you every generation.

---

## 4b. Nesterov blend fix (added after the PyTorch reference review)

The Keras port computed `update = g + beta*m`. Keller's reference computes
`grad.lerp_(momentum, beta)` = `(1-beta)*g + beta*m`. These are NOT proportional, and
since Newton-Schulz normalises magnitude away, only direction matters — at beta=0.95 the
old form weighted the raw gradient ~20x more heavily against momentum. Fixed at
`muon.py:133`.

LR sweep for the corrected blend (`tmp/exp_l_nesterov_lr.py`, exp_j protocol, 3 seeds):

```
NEW blend @ 1e-3      0.5280 +/- 0.0134    |W| 1.00x
NEW blend @ 5e-3      0.6910 +/- 0.0007    |W| 1.00x
NEW blend @ 1e-2      0.7271 +/- 0.0238    |W| 1.04x   <- beats old blend
NEW blend @ 1e-1      0.6110 +/- 0.0676                (was catastrophic, sd 11x higher)
OLD blend @ 1e-1      0.6918 +/- 0.0062                (previous best)
```

**The corrected blend needs ~10x lower lr and then wins** (+0.035, paired +0.061 / +0.051
/ -0.006, about 1.2 sigma — suggestive, not conclusive). Variance is monotone in lr
(0.0007 -> 0.0238 -> 0.0676), which is what identified the earlier failure at 1e-1 as
over-driving rather than a bad direction.

The `(1-beta)` factor does not shrink the step — NS divides by the Frobenius norm first,
so the applied update norm is identical either way. The lower lr is needed because a
momentum-dominated direction makes consecutive steps far more correlated, so displacement
accumulates linearly rather than as sqrt(N).

**Not bracketed:** 1e-2 is the top of the grid and still climbing. True optimum is between
1e-2 and 1e-1.

**Consequence for Connect 4:** the `3e-3` base lr is now doubly stale — it was tuned with
the old blend AND with `use_orthograd: True`. Re-check it before a long run.

### Orthograd re-tested under the corrected blend at a stable lr (`tmp/exp_m_ortho_at_1e2.py`)

```
muon @ 1e-2          0.7271 +/- 0.0237   |W| 1.04x
ortho(muon) @ 1e-2   0.6925 +/- 0.0198   |W| 0.98x
PAIRED: -0.0444  -0.0298  -0.0297   mean -0.0346  sd 0.0069  -> HURTS (5 sigma)
```

This supersedes the "no effect" verdicts in §1. Those comparisons were run at lr 1e-1,
where both arms were unstable (paired sd 0.0517); at a stable lr the paired sd drops to
0.0069 and the penalty becomes unambiguous. Orthograd costs ~3.5 accuracy points in
exchange for a ~6% reduction in weight norm growth (1.04x -> 0.98x).

Conclusion unchanged and now firmly evidenced: **`use_orthograd: False`.**

---

## 5. Caveats

- CIFAR runs are 5–6 epochs on an unconverged model. Run-to-run noise on test accuracy is
  ~3–5 points even with a seeded pipeline; the same config scored 0.7728 and 0.6920 in two
  different scripts before pairing was introduced. Only differences much larger than that
  (the 25-point reshape result) are safe.
- `|W| growth`, `train acc` and the `cos` measurements are far more stable than test
  accuracy and carry most of the mechanistic conclusions.
- Connect 4 experiments used 6,144 positions and 144 optimizer steps against ~150k and
  ~600 in a real generation. Direction is trustworthy, absolute lr values are not.
- Validation loss is not playing strength. AlphaZero quality compounds across generations.

---

## 6. What to do next

1. `use_orthograd: False`.
2. Re-check the Connect 4 base lr at realistic training length — the 3e-3 came from a
   144-step proxy, and CIFAR wanted 1e-1 for Muon.
3. Rerun Connect 4 from generation 0. Bugs 1 and 2 corrupted the data and the input
   representation, so the existing checkpoints are built on wrong foundations.
4. Watch `mean top-move prob` in the policy targets (`tmp/` has the script). It was 0.713
   at gen 9, meaning the search hedged over 2.53 of 5.5 legal moves — about 7 plies of
   sight at 675 sims. Lower `c_puct` should sharpen it; if it does not, that is the thing
   limiting Connect 4, not the optimizer.
