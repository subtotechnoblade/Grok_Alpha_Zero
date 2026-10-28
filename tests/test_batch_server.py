"""Batched inference server (R9, R10). python -m unittest tests.test_batch_server"""
import multiprocessing as mp
import os
import unittest

import numpy as np

from Client_Server import Batch_Server, Batched_Session, create_wakeups
from Transport import ChannelSpec, ClientEnd, create_slots, release_slots, COUNT, SEQ, STATE, REQUEST, RESPONSE, RESP_SEQ
from tests.fake_net import FakeSession
from tests.mcts_scenarios import ROOT

BOARD = (15, 15, 4)
K = 8
SPEC = ChannelSpec({"inputs": (BOARD, np.int8)}, {"policy": ((225,), np.float32), "value": ((1,), np.float32)}, K)
MODEL = os.path.join(ROOT, "Gomoku", "Grok_Zero_Train", "2", "model.onnx")


def boards(rng, n):
    return rng.integers(-1, 2, (n, *BOARD)).astype(np.int8)


def fake():
    return FakeSession(BOARD, 225, seed=5)


def post(slot_client: ClientEnd, rows):
    """Writes a request into a slot without blocking, the way ClientEnd.request does."""
    n = len(rows)
    slot_client.in_views["inputs"][:n] = rows
    slot_client.seq += 1
    slot_client.header[COUNT] = n
    slot_client.header[SEQ] = slot_client.seq
    slot_client.header[STATE] = REQUEST


class PaddingEngine:
    """Pads to a multiple of 8 and trims back, like BucketedEngine."""

    def __init__(self, engine):
        self.engine = engine
        self.seen = []

    def run(self, output_names, input_feed):
        x = input_feed["inputs"]
        n = len(x)
        padded = np.zeros(((n + 7) // 8 * 8, *x.shape[1:]), x.dtype)
        padded[:n] = x
        self.seen.append(len(padded))
        return [o[:n] for o in self.engine.run(output_names, {"inputs": padded})]


class TestRouting(unittest.TestCase):
    def route(self, engine, counts, seed=0):
        rng = np.random.default_rng(seed)
        slots = create_slots(SPEC, len(counts))
        clients = [ClientEnd(SPEC, shm) for shm in slots]
        try:
            server = Batch_Server(SPEC, slots, engine)
            sent = {}
            for i, n in enumerate(counts):
                if n:
                    sent[i] = boards(rng, n)
                    post(clients[i], sent[i])
            ran = server.step()
            self.assertEqual(ran, sum(counts))
            reference = fake()
            for i, client in enumerate(clients):
                if i not in sent:
                    self.assertEqual(client.header[STATE], 0, "an idle slot was touched")
                    continue
                self.assertEqual(client.header[STATE], RESPONSE)
                self.assertEqual(client.header[RESP_SEQ], client.seq)
                n = counts[i]
                p, v = reference.run(["policy", "value"], {"inputs": sent[i].astype(np.float32)})
                self.assertTrue(np.array_equal(client.out_views["policy"][:n], p), f"client {i} policy")
                self.assertTrue(np.array_equal(client.out_views["value"][:n], v), f"client {i} value")
            self.assertEqual(server.step(), 0, "a served slot was served twice")
            return server
        finally:
            for c in clients:
                c.close()
            release_slots(slots)

    def test_uneven_counts_and_idle_slots(self):
        self.route(fake(), [1, K, 3, 0, 5, 0, 2])

    def test_full_capacity(self):
        server = self.route(fake(), [K] * 6)
        self.assertEqual(server.rows, 6 * K)
        self.assertEqual(server.capacity, 6 * K)

    def test_single_row(self):
        self.route(fake(), [1])

    def test_padding_engine(self):
        engine = PaddingEngine(fake())
        self.route(engine, [3, 1, 7])
        self.assertEqual(engine.seen, [16])

    def test_many_random_layouts(self):
        rng = np.random.default_rng(9)
        for trial in range(60):
            counts = [int(c) for c in rng.integers(0, K + 1, int(rng.integers(1, 13)))]
            if sum(counts) == 0:
                counts[0] = 1
            with self.subTest(trial=trial, counts=counts):
                self.route(fake(), counts, seed=trial)


@unittest.skipUnless(os.path.exists(MODEL), "no trained Gomoku model")
class TestRealModelCPU(unittest.TestCase):
    def test_matches_direct_run(self):
        import onnxruntime as rt
        sess = rt.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(3)
        counts = [2, 5, 1, 8]
        slots = create_slots(SPEC, len(counts))
        clients = [ClientEnd(SPEC, shm) for shm in slots]
        try:
            server = Batch_Server(SPEC, slots, sess)
            sent = [boards(rng, n) for n in counts]
            for c, rows in zip(clients, sent):
                post(c, rows)
            server.step()
            concat = np.concatenate(sent).astype(np.float32)
            p_all, v_all = sess.run(["policy", "value"], {"inputs": concat})
            start = 0
            for c, rows, n in zip(clients, sent, counts):
                self.assertTrue(np.array_equal(c.out_views["policy"][:n], p_all[start:start + n]))
                self.assertTrue(np.array_equal(c.out_views["value"][:n], v_all[start:start + n]))
                p_own, v_own = sess.run(["policy", "value"], {"inputs": rows.astype(np.float32)})
                with np.errstate(under="ignore"):
                    self.assertTrue(np.allclose(c.out_views["policy"][:n], p_own, rtol=1e-5, atol=1e-7))
                    self.assertTrue(np.allclose(c.out_views["value"][:n], v_own, rtol=1e-5, atol=1e-6))
                start += n
        finally:
            for c in clients:
                c.close()
            release_slots(slots)


def server_process(slots, request_ready, response_events, stop, stats):
    server = Batch_Server(SPEC, slots, fake(), request_ready, response_events)
    max_batch = 0
    while not stop.is_set():
        ran = server.step()
        max_batch = max(max_batch, ran)
        if ran == 0:
            request_ready.wait(0.01)
    stats.put((server.batches, server.rows, max_batch))


def client_process(i, shm, request_ready, response_ready, n_requests, results):
    rng = np.random.default_rng(100 + i)
    session = Batched_Session(SPEC, shm, request_ready, response_ready)
    reference = fake()
    bad = 0
    for _ in range(n_requests):
        rows = boards(rng, int(rng.integers(1, K + 1)))
        p, v = session.run(["policy", "value"], {"inputs": rows})
        rp, rv = reference.run(["policy", "value"], {"inputs": rows.astype(np.float32)})
        bad += not (np.array_equal(p, rp) and np.array_equal(v, rv))
    results.put((i, bad))


class TestMultiProcess(unittest.TestCase):
    def test_stress(self):
        workers, n_requests = 6, 150
        slots = create_slots(SPEC, workers)
        request_ready, response_events = create_wakeups(workers)
        stop, stats, results = mp.Event(), mp.Queue(), mp.Queue()
        server = mp.Process(target=server_process, args=(slots, request_ready, response_events, stop, stats))
        procs = [mp.Process(target=client_process, args=(i, slots[i], request_ready, response_events[i],
                                                          n_requests, results)) for i in range(workers)]
        try:
            server.start()
            for p in procs:
                p.start()
            outcome = dict(results.get(timeout=300) for _ in procs)
            stop.set()
            batches, rows, max_batch = stats.get(timeout=60)
            self.assertEqual(outcome, {i: 0 for i in range(workers)})
            self.assertLess(batches, workers * n_requests, "requests were never combined")
            self.assertGreater(max_batch, K, "no batch held rows from more than one client")
            print(f"\n  stress: {workers * n_requests} requests in {batches} engine calls, "
                  f"{rows / batches:.1f} rows/call, largest {max_batch}")
        finally:
            stop.set()
            for p in procs + [server]:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
            release_slots(slots)


def mcts_server(slots, request_ready, response_events, stop):
    server = Batch_Server(SPEC, slots, fake(), request_ready, response_events)
    while not stop.is_set():
        if server.step() == 0:
            request_ready.wait(0.01)


class TestMCTSEndToEnd(unittest.TestCase):
    def search(self, session):
        from Gomoku.game import Gomoku
        from MCTS import MCTS
        from tests.mcts_scenarios import snapshot
        np.random.seed(4)
        game = Gomoku()
        game.do_action((7, 7))
        mcts = MCTS(game, session, c_puct_init=2.0, fpu_reduction=0.2, use_dirichlet=True, dirichlet_alpha=0.3,
                    tau=1.0, leaves_per_batch=K)
        records = []
        for _ in range(3):
            move, probs = mcts.run(iteration_limit=300, use_bar=False)
            records.append(snapshot(mcts, move, probs))
            game.do_action(move)
            mcts.prune_tree(move)
        return records

    def test_server_search_equals_local_search(self):
        from Session_Cache import Cache_Wrapper
        from tests.test_mcts_equivalence import assert_same
        slots = create_slots(SPEC, 1)
        (request_ready, response_events), stop = create_wakeups(1), mp.Event()
        server = mp.Process(target=mcts_server, args=(slots, request_ready, response_events, stop))
        server.start()
        try:
            for cached in (False, True):
                with self.subTest(cached=cached):
                    remote = Batched_Session(SPEC, slots[0], request_ready, response_events[0])
                    local = fake()
                    if cached:
                        remote, local = Cache_Wrapper(remote, 4), Cache_Wrapper(local, 4)
                    a, b = self.search(local), self.search(remote)
                    assert_same(self, {"records": a, "cache_stats": None}, {"records": b, "cache_stats": None},
                                f"cached={cached}")
                    remote_client = remote.session.client if cached else remote.client
                    remote_client.close()
        finally:
            stop.set()
            server.join(timeout=30)
            if server.is_alive():
                server.terminate()
            release_slots(slots)


if __name__ == "__main__":
    unittest.main()
