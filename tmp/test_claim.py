"""Race test for the claim counter used in Self_Play.self_play_task.

The failure mode being tested: if the read and the increment are not under the SAME
lock acquisition, two workers can both observe the last index and the generation
plays MORE games than requested (or, with a different slip, fewer).

Mimics the real loop exactly, with jittered work so workers finish at different
rates -- which is the whole point of the change.
"""
import multiprocessing as mp
import random
import sys


def worker(claimed, total, done, wid):
    """Byte-for-byte the same claim protocol as the real worker loop."""
    n = 0
    rng = random.Random(wid)
    while True:
        with claimed.get_lock():
            if claimed.value >= total:
                break
            claimed.value += 1
        # stand-in for Self_Play(...).play(): deliberately uneven, like real games
        s = 0
        for _ in range(rng.randint(200, 4000)):
            s += 1
        n += 1
    done[wid] = n


def run(total, workers):
    claimed = mp.Value("i", 0)
    done = mp.Array("i", [0] * workers)
    procs = [mp.Process(target=worker, args=(claimed, total, done, i))
             for i in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    per = list(done)
    return claimed.value, sum(per), per


if __name__ == "__main__":
    print(f"{'total':>7} {'workers':>8} {'counter':>8} {'games played':>13} "
          f"{'exact?':>7} {'min/max per worker':>20}")
    ok = True
    for total, workers in ((500, 10), (500, 16), (7, 10), (1, 10), (0, 4), (997, 13)):
        counter, played, per = run(total, workers)
        exact = (played == total) and (counter == total)
        ok &= exact
        print(f"{total:>7} {workers:>8} {counter:>8} {played:>13} "
              f"{'YES' if exact else 'NO !!':>7} "
              f"{f'{min(per)} / {max(per)}':>20}")
    print()
    print("PASS: exactly the requested number of games, no overshoot"
          if ok else "FAIL: game count did not match")
    sys.exit(0 if ok else 1)
