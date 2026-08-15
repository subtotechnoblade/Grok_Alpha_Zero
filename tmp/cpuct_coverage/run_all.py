"""Driver: acquire tmp/.resource_locks/cpu.lock, run the c_puct self-play sweep with
<= WORKERS concurrent processes, release the lock no matter what.

usage: run_all.py [games_per_batch] [workers] [shard]
"""
import os
import sys
import time
import subprocess
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(r"C:\Users\TryHarder\Desktop\Pythoncode\Grok_Zero")
RUN = ROOT / "tmp/cpuct_coverage/run1"
LOCKDIR = ROOT / "tmp/.resource_locks"
LOCK = LOCKDIR / "cpu.lock"
PY = r"C:/Users/TryHarder/anaconda3/envs/tensorflow_updated/python.exe"
COLLECT = ROOT / "tmp/cpuct_coverage/sp_collect.py"
MODEL = ROOT / "tmp/run3_archive_gen0-15_bottleneck_preweightfix/12/model.onnx"
CPUCTS = [1.5, 2.5, 3.5, 5.0]
AGENT = "cpuct-coverage-experiment"
STALE = 2 * 3600


def acquire(poll=60, max_wait=6 * 3600):
    LOCKDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        if LOCK.exists():
            age = time.time() - LOCK.stat().st_mtime
            if age < STALE:
                if time.time() - t0 > max_wait:
                    raise SystemExit(f"cpu.lock held for {max_wait}s, giving up")
                print(f"cpu.lock held ({age / 60:.1f} min old) by:\n"
                      f"{LOCK.read_text().strip()}\nwaiting {poll}s...", flush=True)
                time.sleep(poll)
                continue
            print(f"cpu.lock is STALE ({age / 3600:.1f} h) - taking it", flush=True)
        LOCK.write_text(
            f"agent: {AGENT}\npid: {os.getpid()}\n"
            f"desc: c_puct_init coverage/diversity self-play sweep (inference only, "
            f"no training) in tmp/cpuct_coverage\n"
            f"ts: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        print("cpu.lock ACQUIRED", flush=True)
        return


def release():
    try:
        if LOCK.exists() and AGENT in LOCK.read_text():
            LOCK.unlink()
            print("cpu.lock RELEASED", flush=True)
    except Exception as e:  # never let cleanup mask the real error
        print(f"lock release failed: {e}", flush=True)


def touch():
    try:
        LOCK.touch()
    except Exception:
        pass


def run_job(job):
    cp, batch, w, games, seed = job
    tag = f"cp{str(cp).replace('.', 'p')}_b{batch}_w{w}"
    out = RUN / f"{tag}.npz"
    if out.exists():
        return f"skip {tag}"
    cmd = [PY, str(COLLECT), str(cp), str(games), str(seed), str(out), str(MODEL), "375"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        return f"FAIL {tag}: {r.stderr[-600:]}"
    return f"ok   {tag}: {r.stdout.strip()}"


def main():
    games_per_batch = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    shard = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    RUN.mkdir(parents=True, exist_ok=True)

    jobs = []
    seed = 1000
    n_shards = max(1, games_per_batch // shard)
    per = games_per_batch // n_shards
    # shard-major ordering: if the run has to be cut short, every c_puct has the same
    # number of games rather than the first one having them all
    for w in range(n_shards):
        for batch in ("A", "B"):
            for cp in CPUCTS:
                seed += 1
                jobs.append((cp, batch, w, per, seed))
    print(f"{len(jobs)} shards x {per} games = "
          f"{len(jobs) * per} games total, {workers} concurrent", flush=True)

    acquire()
    t0 = time.time()
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for msg in ex.map(run_job, jobs):
                done += 1
                touch()  # keep the lock fresh so nobody calls it stale
                el = time.time() - t0
                print(f"[{done}/{len(jobs)} {el / 60:.1f}min "
                      f"eta {el / done * (len(jobs) - done) / 60:.1f}min] {msg}", flush=True)
    finally:
        release()
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
