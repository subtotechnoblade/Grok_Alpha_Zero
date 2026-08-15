"""Value-target sign audit + length-42 mislabel census + augmentation consistency.

Replays games independently through Connect4 (game rules) and checks the stored
value targets:  target = 0.8*z + 0.2*q, q in [-1,1]  =>  sign(target) == sign(z)
for decisive games (|0.8 z| > |0.2 q|), |target| <= 0.2 for draws.

mover(t) = -1 for even t, +1 for odd t. In a decisive game the winner is the
mover of the last stored ply (the game ends on that move).

Also: Self_Play.py:142-143 overwrites winner=0 whenever actions_count hits
max_actions (42) - even when the 42nd move WINS. Census: every stored game of
length 42, replay the forced last move, see if it wins, and what label it got.
"""
import sys, random
import numpy as np
import h5py as h5

sys.path.insert(0, ".")
from Connect4.game import Connect4

BASE = "Connect4/Grok_Zero_Train"
rng = random.Random(1234)

def moves_from_states(boards):
    """Derive the move sequence from consecutive stored states (channel 3 = board before move t)."""
    L = boards.shape[0]
    moves = []
    for t in range(L - 1):
        b0 = boards[t][:, :, 3].astype(np.int16)
        b1 = boards[t + 1][:, :, 3].astype(np.int16)
        d = np.argwhere(b1 - b0 != 0)
        assert len(d) == 1, f"ply {t}: {len(d)} cells changed"
        r, c = d[0]
        expected_mover = -1 if t % 2 == 0 else 1
        assert b1[r, c] == expected_mover, f"ply {t}: piece {b1[r,c]} != mover {expected_mover}"
        moves.append(int(c))
    return moves

def replay(moves):
    """Replay through the real game class; return list of check_win() results after each move."""
    g = Connect4()
    results = []
    for m in moves:
        g.do_action(m)
        results.append(g.check_win())
    return g, results

# ---------- Part 1: random 20-game sign audit ----------
print("=" * 70)
print("PART 1: value-sign audit, 2 random games per generation")
print("=" * 70)
bad = 0
for gen in range(10):
    with h5.File(f"{BASE}/{gen}/Self_Play_Data.h5", "r") as f:
        num_games = int(f["game_stats"][2])
        augs = ((len(f.keys()) - 1) // 3) // num_games
        for game_id in rng.sample(range(num_games), 2):
            boards = f[f"boards_{game_id * augs}"][:]
            vals = f[f"values_{game_id * augs}"][:].ravel()
            L = boards.shape[0]
            moves = moves_from_states(boards)
            g, results = replay(moves)
            # game must not be over before the last stored ply
            assert all(r == -2 for r in results), f"gen{gen} game{game_id}: early game end in replay"

            if L < 42:
                true_winner = -1 if (L - 1) % 2 == 0 else 1  # mover of last ply
                label = "WIN"
            else:
                # forced last move: the single remaining empty cell's column
                empty = np.argwhere(g.board == 0)
                assert len(empty) == 1
                g.do_action(int(empty[0][1]))
                w = g.check_win()
                true_winner = int(w)
                label = "WIN@42" if w != 0 else "DRAW"

            ok = True
            if true_winner != 0:
                # sign must alternate and final ply (mover=winner) must be positive
                signs_ok = all(vals[t] * vals[t + 1] < 0 for t in range(L - 1))
                final_ok = vals[-1] > 0.6 - 1e-6
                # mover of ply t won iff mover(t)==true_winner => v_t>0
                persp_ok = all((vals[t] > 0) == ((-1 if t % 2 == 0 else 1) == true_winner)
                               for t in range(L))
                ok = signs_ok and final_ok and persp_ok
                detail = f"alternates={signs_ok} final={vals[-1]:+.3f} perspective={persp_ok}"
            else:
                draw_ok = np.all(np.abs(vals) <= 0.2 + 1e-6)
                ok = bool(draw_ok)
                detail = f"all |v|<=0.2: {draw_ok} (max |v|={np.abs(vals).max():.3f})"
            if not ok:
                bad += 1
            print(f"gen{gen} game{game_id:3d} len={L:2d} {label:7s} true_winner={true_winner:+d} "
                  f"stored v[0]={vals[0]:+.3f} v[-1]={vals[-1]:+.3f} -> {'OK' if ok else 'MISMATCH'} ({detail})")
print(f"\nsign audit mismatches: {bad}/20")

# ---------- Part 2: length-42 census ----------
print()
print("=" * 70)
print("PART 2: length-42 games - was the 42nd (forced) move a win, and how was it labeled?")
print("=" * 70)
tot42 = totwin42 = totmislab = totgames = 0
for gen in range(10):
    n42 = win42 = mislab = 0
    with h5.File(f"{BASE}/{gen}/Self_Play_Data.h5", "r") as f:
        num_games = int(f["game_stats"][2])
        augs = ((len(f.keys()) - 1) // 3) // num_games
        for game_id in range(num_games):
            boards = f[f"boards_{game_id * augs}"]
            L = boards.shape[0]
            if L != 42:
                continue
            n42 += 1
            moves = moves_from_states(boards[:])
            g, results = replay(moves)
            empty = np.argwhere(g.board == 0)
            g.do_action(int(empty[0][1]))
            w = g.check_win()
            if w != 0:
                win42 += 1
                vals = f[f"values_{game_id * augs}"][:].ravel()
                # if stored as draw, z was zeroed: |v| <= 0.2 everywhere
                if np.abs(vals).max() <= 0.2 + 1e-6:
                    mislab += 1
    stats42 = f"gen {gen}: games_len42={n42:3d}  won_on_move42={win42:3d}  stored_as_draw={mislab:3d}"
    print(stats42)
    tot42 += n42; totwin42 += win42; totmislab += mislab; totgames += num_games
print(f"\nTOTAL: {tot42} len-42 games, {totwin42} were wins on move 42, "
      f"{totmislab} of those are stored with draw (z=0) targets")

# ---------- Part 3: augmentation consistency ----------
print()
print("=" * 70)
print("PART 3: mirror augmentation consistency (random 30 (game, ply) picks)")
print("=" * 70)
bad_aug = 0
for _ in range(30):
    gen = rng.randrange(10)
    with h5.File(f"{BASE}/{gen}/Self_Play_Data.h5", "r") as f:
        num_games = int(f["game_stats"][2])
        augs = ((len(f.keys()) - 1) // 3) // num_games
        game_id = rng.randrange(num_games)
        b0 = f[f"boards_{game_id * augs}"][:]
        b1 = f[f"boards_{game_id * augs + 1}"][:]
        p0 = f[f"policies_{game_id * augs}"][:]
        p1 = f[f"policies_{game_id * augs + 1}"][:]
        v0 = f[f"values_{game_id * augs}"][:]
        v1 = f[f"values_{game_id * augs + 1}"][:]
        t = rng.randrange(b0.shape[0])
        ok = (np.array_equal(b1[t], b0[t][:, ::-1, :])
              and np.array_equal(p1[t], p0[t][::-1])
              and np.array_equal(v1[t], v0[t]))
        if not ok:
            bad_aug += 1
            print(f"  MISMATCH gen{gen} game{game_id} ply{t}")
print(f"augmentation mismatches: {bad_aug}/30")
