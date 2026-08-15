"""Diversity / cross-batch coverage / search-behaviour analysis over sp_collect npz shards.

usage: analyze.py <run_dir>
  expects shards named cp<c_puct>_b<A|B>_w<worker>.npz
"""
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BANDS = [(0, 5), (6, 9), (10, 13), (14, 17), (18, 99)]
BAND_NAMES = ["0-5", "6-9", "10-13", "14-17", "18+"]

# ints columns produced by sp_collect
PVD, EXPLORE, NLEG, NWIN, PWIN, NSAFE, PSAFE = range(7)


def band_of(ply):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= ply <= hi:
            return i
    return len(BANDS) - 1


class Batch:
    def __init__(self):
        self.ply = []
        self.skey = []
        self.bkey = []
        self.effb = []
        self.top1 = []
        self.pvd = []
        self.expl = []
        self.tac = []          # (nwin, pwin, nsafe, pwin_safe, nleg)
        self.games = []        # (winner, length)
        self.wall = 0.0
        self.ngames = 0

    def add(self, npz):
        d = np.load(npz)
        gp = d["gid_ply"]
        self.ply.append(gp[:, 1])
        self.skey.append(d["keys"][:, 1])
        self.bkey.append(d["keys"][:, 0])
        f = d["floats"]
        self.effb.append(f[:, 0])
        self.top1.append(f[:, 1])
        it = d["ints"]
        self.pvd.append(it[:, PVD])
        self.expl.append(it[:, EXPLORE])
        self.tac.append(it[:, [NWIN, PWIN, NSAFE, PSAFE, NLEG]])
        self.games.append(d["games"][:, 1:])
        m = d["meta"]
        self.wall += m[4]
        self.ngames += int(m[3])
        self.c_puct = float(m[0])

    def finish(self):
        self.ply = np.concatenate(self.ply)
        self.skey = np.concatenate(self.skey)
        self.bkey = np.concatenate(self.bkey)
        self.effb = np.concatenate(self.effb)
        self.top1 = np.concatenate(self.top1)
        self.pvd = np.concatenate(self.pvd)
        self.expl = np.concatenate(self.expl)
        self.tac = np.concatenate(self.tac)
        self.games = np.concatenate(self.games)
        self.band = np.array([band_of(p) for p in self.ply])
        return self


def load(run_dir):
    out = defaultdict(lambda: defaultdict(Batch))
    for p in sorted(Path(run_dir).glob("cp*_b*_w*.npz")):
        m = re.match(r"cp([0-9p.]+)_b([AB])_w\d+\.npz", p.name)
        cp = m.group(1).replace("p", ".")
        out[float(cp)][m.group(2)].add(str(p))
    return {cp: {b: v.finish() for b, v in d.items()} for cp, d in out.items()}


def diversity_table(bt: Batch, n_games=None):
    """rows: band -> total, distinct, dedup%, distinct/game, distinct/100games"""
    ng = n_games if n_games is not None else bt.ngames
    rows = []
    for bi, name in enumerate(BAND_NAMES):
        m = bt.band == bi
        tot = int(m.sum())
        dis = len(np.unique(bt.skey[m])) if tot else 0
        rows.append((name, tot, dis, 100.0 * (1 - dis / tot) if tot else 0.0,
                     dis / ng, 100.0 * dis / ng))
    return rows


def coverage(a: Batch, b: Batch):
    """fraction of batch B's positions found in batch A, weighted by occurrence in B,
    and the unweighted distinct-position version."""
    rows = []
    for bi, name in enumerate(BAND_NAMES):
        ma = a.band == bi
        mb = b.band == bi
        setA = set(a.skey[ma].tolist())
        kb = b.skey[mb]
        if len(kb) == 0:
            rows.append((name, 0, 0.0, 0.0))
            continue
        hit = np.fromiter((k in setA for k in kb.tolist()), dtype=bool, count=len(kb))
        w = 100.0 * hit.mean()                       # occurrence-weighted
        ub = np.unique(kb)
        uh = np.fromiter((k in setA for k in ub.tolist()), dtype=bool, count=len(ub))
        rows.append((name, len(kb), w, 100.0 * uh.mean()))
    return rows


def search_table(bt: Batch):
    rows = []
    for bi, name in enumerate(BAND_NAMES):
        m = bt.band == bi
        if not m.any():
            rows.append((name, 0, 0, 0))
            continue
        rows.append((name, bt.effb[m].mean(), bt.top1[m].mean(), bt.pvd[m].mean()))
    return rows


def quality(bt: Batch):
    w = bt.games[:, 0]
    L = bt.games[:, 1]
    nwin, pwin, nsafe, psafe, nleg = (bt.tac[:, i] for i in range(5))
    greedy = bt.expl == 0
    has_win = (nwin > 0) & greedy
    must_block = (nwin == 0) & (nsafe > 0) & (nsafe < nleg) & greedy
    lost_pos = (nwin == 0) & (nsafe == 0) & greedy
    return {
        "n_games": len(w),
        "p-1": 100.0 * (w == -1).mean(),
        "draw": 100.0 * (w == 0).mean(),
        "p+1": 100.0 * (w == 1).mean(),
        "len": L.mean(),
        "n_haswin": int(has_win.sum()),
        "win_taken": 100.0 * pwin[has_win].mean() if has_win.any() else float("nan"),
        "n_block": int(must_block.sum()),
        "block_taken": 100.0 * psafe[must_block].mean() if must_block.any() else float("nan"),
        "n_lost": int(lost_pos.sum()),
        "s_per_game": bt.wall / bt.ngames,
    }


def main():
    run = sys.argv[1]
    data = load(run)
    cps = sorted(data)

    print("=" * 100)
    print("1. POSITION DIVERSITY  (key = byte-identical get_input_state(), i.e. what dedup_average merges)")
    print("=" * 100)
    for cp in cps:
        for bl in sorted(data[cp]):
            bt = data[cp][bl]
            print(f"\n-- c_puct={cp}  batch {bl}  ({bt.ngames} games, {len(bt.ply)} positions) --")
            print(f"{'band':>6} {'positions':>10} {'distinct':>9} {'dedup%':>7} "
                  f"{'dist/game':>10} {'dist/100g':>10}")
            for name, tot, dis, dd, dpg, dp100 in diversity_table(bt):
                print(f"{name:>6} {tot:>10} {dis:>9} {dd:>7.1f} {dpg:>10.2f} {dp100:>10.1f}")

    print()
    print("=" * 100)
    print("2. CROSS-BATCH COVERAGE   fraction of batch B positions present in batch A")
    print("   (occ-weighted = the direct analogue of the train->deployment number)")
    print("=" * 100)
    print(f"{'band':>6} " + " ".join(f"{'cp=' + str(cp):>16}" for cp in cps))
    cov = {cp: coverage(data[cp]["A"], data[cp]["B"]) for cp in cps if "B" in data[cp]}
    for i, name in enumerate(BAND_NAMES):
        cells = []
        for cp in cps:
            if cp in cov:
                _, n, w, u = cov[cp][i]
                cells.append(f"{w:>7.1f}% ({u:>4.1f}%)")
            else:
                cells.append(f"{'-':>16}")
        print(f"{name:>6} " + " ".join(f"{c:>16}" for c in cells))
    print("   format: occurrence-weighted% (distinct-position%)")

    print()
    print("=" * 100)
    print("3. SEARCH BEHAVIOUR AT THE ROOT   (batch A+B pooled)")
    print("=" * 100)
    for metric, idx in (("eff. branching exp(H)", 1), ("top-1 visit share", 2), ("PV depth", 3)):
        print(f"\n-- {metric} --")
        print(f"{'band':>6} " + " ".join(f"{'cp=' + str(cp):>9}" for cp in cps))
        tabs = {}
        for cp in cps:
            merged = Batch()
            for bl in data[cp]:
                bt = data[cp][bl]
                merged.ply.append(bt.ply)
                merged.skey.append(bt.skey)
                merged.bkey.append(bt.bkey)
                merged.effb.append(bt.effb)
                merged.top1.append(bt.top1)
                merged.pvd.append(bt.pvd)
                merged.expl.append(bt.expl)
                merged.tac.append(bt.tac)
                merged.games.append(bt.games)
                merged.ngames += bt.ngames
                merged.wall += bt.wall
            tabs[cp] = search_table(merged.finish())
        for i, name in enumerate(BAND_NAMES):
            print(f"{name:>6} " + " ".join(f"{tabs[cp][i][idx]:>9.2f}" for cp in cps))

    print()
    print("=" * 100)
    print("4. GAME QUALITY   (batch A+B pooled; tactics measured at tau=0 plies only)")
    print("=" * 100)
    hdr = ["n_games", "p-1 %", "draw %", "p+1 %", "avg len", "haswin n", "win taken %",
           "block n", "block taken %", "lost n", "s/game"]
    print(f"{'c_puct':>7} " + " ".join(f"{h:>13}" for h in hdr))
    for cp in cps:
        merged = Batch()
        for bl in data[cp]:
            bt = data[cp][bl]
            for attr in ("ply", "skey", "bkey", "effb", "top1", "pvd", "expl", "tac", "games"):
                getattr(merged, attr).append(getattr(bt, attr))
            merged.ngames += bt.ngames
            merged.wall += bt.wall
        q = quality(merged.finish())
        print(f"{cp:>7} " + " ".join(f"{v:>13.2f}" if isinstance(v, float) else f"{v:>13}"
                                     for v in [q["n_games"], q["p-1"], q["draw"], q["p+1"],
                                               q["len"], q["n_haswin"], q["win_taken"],
                                               q["n_block"], q["block_taken"], q["n_lost"],
                                               q["s_per_game"]]))

    print()
    print("=" * 100)
    print("5. COVERAGE vs NUMBER OF GAMES   (A truncated at game boundaries; B held full)")
    print("=" * 100)
    for cp in cps:
        if "B" not in data[cp]:
            continue
        a, b = data[cp]["A"], data[cp]["B"]
        starts = np.where(a.ply == 0)[0]          # one entry per game played in A
        print(f"
-- c_puct={cp} --")
        print(f"{'A games':>8} " + " ".join(f"{n:>9}" for n in BAND_NAMES))
        for frac in (0.125, 0.25, 0.5, 1.0):
            ng = max(1, int(round(len(starts) * frac)))
            end = len(a.ply) if ng >= len(starts) else int(starts[ng])
            skey, band = a.skey[:end], a.band[:end]
            cells = []
            for bi in range(len(BAND_NAMES)):
                setA = set(skey[band == bi].tolist())
                kb = b.skey[b.band == bi]
                cells.append(100.0 * np.fromiter((k in setA for k in kb.tolist()),
                                                 dtype=bool, count=len(kb)).mean()
                             if len(kb) else 0.0)
            print(f"{ng:>8} " + " ".join(f"{c:>8.1f}%" for c in cells))


if __name__ == "__main__":
    main()
