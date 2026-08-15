"""Audit of Client_Server.py: does worker i actually get back f(input_i)?

Runs the REAL Server and Parallelized_Session classes. onnxruntime is stubbed with a
deterministic per-sample function so any cross-wiring between workers shows up as a
mismatch rather than as slightly-wrong logits nobody would notice.

    python tmp/audit_client_server.py
"""
import os, sys, time, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import multiprocessing as mp

# Stub onnxruntime before Client_Server imports it (Server does the import inside __init__,
# so seeding sys.modules is enough, and fork carries it into the server process).
_fake = types.ModuleType("onnxruntime")


class FakeSession:
    def __init__(self, path, providers=None, **kw):
        pass

    def run(self, names, input_feed):
        x = np.asarray(input_feed["inputs"])
        b = x.shape[0]
        s = x.reshape(b, -1).sum(1).astype(np.float32)      # unique per sample by construction
        policy = (s[:, None] + np.arange(7, dtype=np.float32)[None, :]).astype(np.float32)
        value = (s[:, None] * 2.0).astype(np.float32)
        return [policy, value]


_fake.InferenceSession = FakeSession
_fake.SessionOptions = object
_fake.get_available_providers = lambda: []
sys.modules["onnxruntime"] = _fake

from Client_Server import (Parallelized_Session, create_shared_memory,
                           convert_to_single_info, start_server)

SHAPE = (6, 7, 4)
NW, TRIALS = 4, 25


def batched_info():
    return ({"inputs": [[-1, *SHAPE], np.float32]}, {"policy": [-1, 7], "value": [-1, 1]})


def worker(wid, shm, result_q):
    bi, bo = batched_info()
    sess = Parallelized_Session(wid, shm, convert_to_single_info(bi), convert_to_single_info(bo))
    bad = []
    for t in range(TRIALS):
        x = np.full((1, *SHAPE), wid * 1000 + t, dtype=np.float32)
        p, v = sess.run(["policy", "value"], {"inputs": x})
        s = x.sum()
        if not np.allclose(p[0], s + np.arange(7, dtype=np.float32)) or not np.allclose(v[0], s * 2):
            bad.append((t, float(p[0][0]), float(s)))
    result_q.put((wid, len(bad), bad[:3]))


def main():
    bi, bo = batched_info()
    shms = create_shared_memory(bi, bo, NW)
    print(f"shm size: {shms[0].size} bytes = {shms[0].size // 4} float32 slots "
          f"(need 1 flag + {int(np.prod(SHAPE))} inputs)")

    srv_bi, srv_bo = batched_info()          # fresh: Server.__init__ mutates these lists
    server = mp.Process(target=start_server, args=(srv_bi, srv_bo, shms, [], "unused.onnx", 0.001))
    server.start()

    q = mp.Queue()
    jobs = [mp.Process(target=worker, args=(i, shms[i], q)) for i in range(NW)]
    t0 = time.time()
    for j in jobs:
        j.start()
    for j in jobs:
        j.join(timeout=60)

    total_bad = 0
    for _ in range(NW):
        wid, nbad, sample = q.get(timeout=10)
        total_bad += nbad
        print(f"  worker {wid}: {TRIALS - nbad}/{TRIALS} round-trips correct"
              + (f"   MISMATCHES {sample}" if nbad else ""))
    print(f"\n{NW * TRIALS} concurrent round-trips in {time.time()-t0:.2f}s, "
          f"{'ALL CORRECT' if total_bad == 0 else f'{total_bad} WRONG'}")

    server.terminate(); server.join()

    print("\n--- does shm.close() work while Parallelized_Session holds a numpy view? ---")
    bi2, bo2 = batched_info()
    s2 = create_shared_memory(bi2, bo2, 1)[0]
    sess = Parallelized_Session(0, s2, convert_to_single_info(bi2), convert_to_single_info(bo2))
    try:
        s2.close()
        print("  close() OK")
    except BufferError as e:
        print(f"  BufferError: {e}")
        print("  -> self_play_task calls info.close() with the session still alive")
    finally:
        del sess
        try:
            s2.close()
        except Exception:
            pass
        s2.unlink()

    for s in shms:
        try:
            s.close()
        except Exception:
            pass
        s.unlink()


if __name__ == "__main__":
    main()
