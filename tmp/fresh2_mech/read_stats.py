"""Read-only peek at game_stats for every generation. Never writes."""
import h5py as h5

BASE = r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero\Connect4\Grok_Zero_Train"

print(f"{'gen':>4} {'games':>6} {'p-1 win':>8} {'draw':>7} {'p+1 win':>8} {'avg mv':>7} {'max mv':>7}")
for gen in range(13):
    path = rf"{BASE}\{gen}\Self_Play_Data.h5"
    try:
        with h5.File(path, mode="r") as f:
            ma, ta, n, w1, dr, w2 = [int(x) for x in f["game_stats"][:]]
    except OSError as e:
        print(f"{gen:>4}  skipped ({e.__class__.__name__})")
        continue
    if n == 0:
        print(f"{gen:>4}  0 games")
        continue
    print(f"{gen:>4} {n:>6} {w1/n:>8.3f} {dr/n:>7.3f} {w2/n:>8.3f} {ta/n:>7.1f} {ma:>7}")
