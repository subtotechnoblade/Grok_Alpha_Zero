"""Disk cache vs in-memory dict: what the Session_Cache rewrite actually bought.

Uses the OLD implementation verbatim from git (HEAD:Session_Cache.py) against the new
one, on Connect4-sized keys (1,6,7,4) float32 = 672 bytes, with a Zipf-like access
pattern rather than hammering one key - a few hot opening positions plus a long tail,
which is what MCTS actually produces.

Three things measured:
  A. single process per-operation latency, diskcache on native Linux fs AND on /mnt/c
     (the project lives on /mnt/c, so that is the real path the old code used)
  B. 8-process contention, which is the real deployment shape
  C. the finished_lookup latch, i.e. what the old code did in practice vs in principle

    python tmp/bench_cache.py
"""
import os, sys, time, shutil, tempfile, statistics
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diskcache import Cache
from Session_Cache import Cache_Wrapper as NewWrapper

KEY_SHAPE = (1, 6, 7, 4)
KEY_BYTES = int(np.prod(KEY_SHAPE)) * 4
N_DISTINCT = 20_000
N_OPS = 20_000
INFERENCE_S = 3.0e-4      # measured in tmp/audit_client_server.py: ~0.3ms per round trip


class OldWrapper:
    """Verbatim git HEAD:Session_Cache.py"""

    def __init__(self, session, path, max_cache_depth=2):
        self.session = session
        self.path = path
        self.finished_lookup = False if max_cache_depth > 0 else 0
        self.cache = Cache(path, timeout=100000)
        self.max_cache_depth = max_cache_depth

    def run(self, output_names, input_feed: dict, depth):
        if self.max_cache_depth > 0 and not self.finished_lookup:
            key = np.ascontiguousarray(input_feed["inputs"].flatten()).newbyteorder("little").tobytes()
            outputs = self.cache.get(key)
            if outputs is not None:
                return outputs
            self.finished_lookup = True
        outputs = self.session.run(output_names, input_feed)
        if depth < self.max_cache_depth:
            key = np.ascontiguousarray(input_feed["inputs"].flatten()).newbyteorder("little").tobytes()
            self.cache.set(key, outputs, expire=None)
        return outputs


class FakeSession:
    """Stands in for the NN. Counts calls so we can see the hit rate."""

    def __init__(self):
        self.calls = 0

    def run(self, names, input_feed):
        self.calls += 1
        return [np.zeros((1, 7), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)]


def make_keys(n, seed=0):
    rng = np.random.RandomState(seed)
    # board-like: -1/0/1 planes, distinct by construction via a unique counter plane
    keys = rng.randint(-1, 2, size=(n, *KEY_SHAPE)).astype(np.float32)
    keys[:, 0, 0, 0, 0] = np.arange(n, dtype=np.float32)
    return keys


def zipf_access(n_distinct, n_ops, seed=1):
    """Heavy head, long tail. Returns indices into the key table."""
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, n_distinct + 1)
    p = 1.0 / ranks ** 0.9
    p /= p.sum()
    return rng.choice(n_distinct, size=n_ops, p=p)


def bench(wrapper, keys, order, label):
    sess = wrapper.session
    before = sess.calls
    t0 = time.perf_counter()
    for i in order:
        wrapper.run(["policy", "value"], {"inputs": keys[i]}, depth=0)
    elapsed = time.perf_counter() - t0
    misses = sess.calls - before
    hits = len(order) - misses
    per_op = elapsed / len(order)
    print(f"  {label:<34} {elapsed:7.3f}s total   {per_op*1e6:8.1f} us/op   "
          f"{hits/len(order):6.1%} hit")
    return per_op, hits / len(order)


def part_a(tmp_native, tmp_mnt):
    print(f"\nA. single process, {N_OPS} ops over {N_DISTINCT} distinct keys "
          f"({KEY_BYTES} B each), zipf access\n")
    keys = make_keys(N_DISTINCT)
    order = zipf_access(N_DISTINCT, N_OPS)

    results = {}
    new = NewWrapper(FakeSession(), max_cache_depth=99, max_entries=10 ** 9)
    results["dict"] = bench(new, keys, order, "new: in-memory dict")

    for name, base in (("diskcache (native linux fs)", tmp_native),
                       ("diskcache (/mnt/c, real path)", tmp_mnt)):
        if base is None:
            print(f"  {name:<34} skipped")
            continue
        path = os.path.join(base, f"c{abs(hash(name))%9999}")
        old = OldWrapper(FakeSession(), path, max_cache_depth=99)
        old.finished_lookup = False
        # neutralise the latch here so this measures STORAGE cost, not the latch.
        # Part C measures the latch separately.
        orig_run = old.run

        def run_no_latch(output_names, input_feed, depth, _o=old):
            _o.finished_lookup = False
            return orig_run(output_names, input_feed, depth)

        old.run = run_no_latch
        results[name] = bench(old, keys, order, "old: " + name)
        old.cache.close()
        shutil.rmtree(path, ignore_errors=True)
    return results


def worker_dict(n_ops, seed, q):
    keys = make_keys(N_DISTINCT)
    order = zipf_access(N_DISTINCT, n_ops, seed)
    w = NewWrapper(FakeSession(), max_cache_depth=99, max_entries=10 ** 9)
    t0 = time.perf_counter()
    for i in order:
        w.run(["policy", "value"], {"inputs": keys[i]}, depth=0)
    q.put(time.perf_counter() - t0)


def worker_disk(n_ops, seed, path, q):
    keys = make_keys(N_DISTINCT)
    order = zipf_access(N_DISTINCT, n_ops, seed)
    w = OldWrapper(FakeSession(), path, max_cache_depth=99)
    orig = w.run
    t0 = time.perf_counter()
    for i in order:
        w.finished_lookup = False
        orig(["policy", "value"], {"inputs": keys[i]}, depth=0)
    q.put(time.perf_counter() - t0)
    w.cache.close()


def part_b(tmp_native):
    n_workers, n_ops = 8, 4000
    print(f"\nB. {n_workers} concurrent workers x {n_ops} ops "
          f"(diskcache shares ONE sqlite db, dicts are per process)\n")
    for label, target, extra in (("new: per-process dicts", worker_dict, ()),
                                 ("old: shared diskcache", worker_disk,
                                  (os.path.join(tmp_native, "shared"),))):
        q = mp.Queue()
        procs = [mp.Process(target=target, args=(n_ops, s, *extra, q))
                 for s in range(n_workers)]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0
        times = [q.get() for _ in range(n_workers)]
        total_ops = n_workers * n_ops
        print(f"  {label:<28} wall {wall:6.2f}s   "
              f"worker mean {statistics.mean(times):5.2f}s   "
              f"{wall/total_ops*1e6:7.1f} us/op aggregate")
        if extra:
            shutil.rmtree(extra[0], ignore_errors=True)


def part_c(tmp_native):
    print("\nC. what the old code did in PRACTICE: the finished_lookup latch\n")
    keys = make_keys(N_DISTINCT)
    order = zipf_access(N_DISTINCT, 5000, seed=3)
    path = os.path.join(tmp_native, "latch")
    old = OldWrapper(FakeSession(), path, max_cache_depth=99)
    t0 = time.perf_counter()
    for i in order:
        old.run(["policy", "value"], {"inputs": keys[i]}, depth=0)
    elapsed = time.perf_counter() - t0
    hits = len(order) - old.session.calls
    print(f"  latch as written    {elapsed:7.3f}s   {hits}/{len(order)} hits "
          f"({hits/len(order):.2%})")
    print(f"  -> after the FIRST miss the latch disables lookups permanently, so every")
    print(f"     later repeat still paid a full inference AND a full sqlite write.")
    old.cache.close()
    shutil.rmtree(path, ignore_errors=True)


def main():
    tmp_native = tempfile.mkdtemp(prefix="bench_native_")
    mnt = os.path.join(ROOT, "tmp", "_bench_mnt")
    os.makedirs(mnt, exist_ok=True)
    try:
        res = part_a(tmp_native, mnt)
        part_b(tmp_native)
        part_c(tmp_native)

        print(f"\nD. cost relative to the inference it avoids "
              f"({INFERENCE_S*1e6:.0f} us per NN round trip)\n")
        for name, (per_op, _) in res.items():
            print(f"  {name:<34} {per_op/INFERENCE_S:6.1%} of one inference")
    finally:
        shutil.rmtree(tmp_native, ignore_errors=True)
        shutil.rmtree(mnt, ignore_errors=True)


if __name__ == "__main__":
    main()
