"""Transport channel. python -m unittest tests.test_transport"""
import multiprocessing as mp
import pickle
import threading
import time
import unittest

import numpy as np

from Transport import (ChannelSpec, ClientEnd, ServerEnd, Wakeup, create_slots, release_slots,
                       IDLE, REQUEST, RESPONSE, STATE, COUNT, SEQ, RESP_SEQ, ALIGN, HEADER_BYTES)

DTYPES = [np.int8, np.uint8, np.int16, np.float16, np.float32, np.float64, np.int64, np.bool_]
SHAPES = [(), (1,), (7,), (15, 15, 4), (3, 2, 5)]


def random_rows(rng, shape, dtype, n):
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return rng.random((n, *shape)) < 0.5
    if dtype.kind == "f":
        with np.errstate(all="ignore"):  # MCTS sets np.seterr(all='raise') for the whole process
            return rng.standard_normal((n, *shape)).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, (n, *shape), endpoint=True, dtype=dtype)


def echo_spec(inputs, max_rows):
    """Outputs echo every input, plus a per-row tag [client, seq, row]."""
    outputs = {f"echo_{name}": spec for name, spec in inputs.items()}
    outputs["tag"] = ((3,), np.int64)
    return ChannelSpec(inputs, outputs, max_rows)


def serve(server: ServerEnd, client_id=0):
    n = server.pending()
    if n == 0:
        return False
    ins, outs = server.inputs(n), server.outputs(n)
    for name, rows in ins.items():
        outs[f"echo_{name}"][:] = rows
    outs["tag"][:, 0] = client_id
    outs["tag"][:, 1] = server.header[SEQ]
    outs["tag"][:, 2] = np.arange(n)
    server.respond()
    return True


class ThreadServer:
    def __init__(self, server, mutate=None):
        self.server, self.mutate = server, mutate
        self.stop = False
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stop:
            if self.server.pending():
                if self.mutate:
                    self.mutate(self.server)
                else:
                    serve(self.server)
            else:
                time.sleep(0)

    def close(self):
        self.stop = True
        self.thread.join()


class Pair:
    def __init__(self, spec, **client_kwargs):
        self.slots = create_slots(spec, 1)
        self.client = ClientEnd(spec, self.slots[0], **client_kwargs)
        self.server = ServerEnd(spec, self.slots[0])

    def close(self):
        self.client.close()
        self.server.close()
        release_slots(self.slots)


class TestLayout(unittest.TestCase):
    def test_random_specs(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            def group(prefix):
                return {f"{prefix}{i}": (tuple(int(d) for d in rng.integers(1, 6, int(rng.integers(0, 4)))),
                                         DTYPES[int(rng.integers(len(DTYPES)))])
                        for i in range(int(rng.integers(1, 4)))}
            spec = ChannelSpec(group("in"), group("out"), int(rng.integers(1, 70)))
            regions = []
            for a in list(spec.inputs.values()) + list(spec.outputs.values()):
                self.assertEqual(a.offset % ALIGN, 0)
                self.assertGreaterEqual(a.offset, HEADER_BYTES)
                regions.append((a.offset, a.offset + a.row_nbytes * spec.max_rows))
            regions.sort()
            for (s0, e0), (s1, e1) in zip(regions, regions[1:]):
                self.assertLessEqual(e0, s1)
            self.assertLessEqual(regions[-1][1], spec.nbytes)

    def test_pickle_and_equality(self):
        spec = echo_spec({"inputs": ((15, 15, 4), np.int8), "mask": ((225,), np.bool_)}, 32)
        clone = pickle.loads(pickle.dumps(spec))
        self.assertEqual(spec, clone)
        self.assertEqual(spec.nbytes, clone.nbytes)
        self.assertEqual([a.offset for a in spec.outputs.values()], [a.offset for a in clone.outputs.values()])
        self.assertNotEqual(spec, echo_spec({"inputs": ((15, 15, 4), np.int8)}, 32))

    def test_bad_specs(self):
        with self.assertRaises(ValueError):
            ChannelSpec({"a": ((2,), np.int8)}, {"b": ((2,), np.int8)}, 0)
        with self.assertRaises(ValueError):
            ChannelSpec({}, {"b": ((2,), np.int8)}, 4)
        with self.assertRaises(ValueError):
            ChannelSpec({"a": ((0,), np.int8)}, {"b": ((2,), np.int8)}, 4)

    def test_slot_too_small(self):
        small = echo_spec({"a": ((4,), np.int8)}, 2)
        big = echo_spec({"a": ((400,), np.float64)}, 64)
        slots = create_slots(small, 1)
        try:
            with self.assertRaises(ValueError):
                ServerEnd(big, slots[0])
        finally:
            release_slots(slots)


class TestRoundTrip(unittest.TestCase):
    def test_every_dtype_and_shape(self):
        rng = np.random.default_rng(1)
        for dtype in DTYPES:
            for shape in SHAPES:
                spec = echo_spec({"x": (shape, dtype), "y": ((2,), np.float32)}, 9)
                pair = Pair(spec)
                srv = ThreadServer(pair.server)
                try:
                    for n in (1, 4, 9):
                        with self.subTest(dtype=np.dtype(dtype).name, shape=shape, n=n):
                            x = random_rows(rng, shape, dtype, n)
                            y = random_rows(rng, (2,), np.float32, n)
                            out = pair.client.request({"x": x, "y": y})
                            self.assertEqual(list(out), ["echo_x", "echo_y", "tag"])
                            self.assertEqual(out["echo_x"].dtype, np.dtype(dtype))
                            self.assertEqual(out["echo_x"].tobytes(), x.tobytes())
                            self.assertEqual(out["echo_y"].tobytes(), y.tobytes())
                            self.assertEqual(out["tag"].shape, (n, 3))
                            self.assertTrue(np.array_equal(out["tag"][:, 2], np.arange(n)))
                            self.assertTrue(np.all(out["tag"][:, 1] == pair.client.seq))
                            self.assertEqual(pair.client.header[STATE], IDLE)
                finally:
                    srv.close()
                    pair.close()

    def test_shrinking_requests_never_return_stale_rows(self):
        spec = echo_spec({"x": ((5,), np.int16)}, 16)
        pair = Pair(spec)
        seen = []
        srv = ThreadServer(pair.server, mutate=lambda s: (seen.append(s.pending()), serve(s)))
        try:
            big = np.full((16, 5), 7, np.int16)
            small = np.full((3, 5), -2, np.int16)
            pair.client.request({"x": big})
            out = pair.client.request({"x": small})
            self.assertEqual(out["echo_x"].shape, (3, 5))
            self.assertTrue(np.all(out["echo_x"] == -2))
            self.assertEqual(seen, [16, 3])
        finally:
            srv.close()
            pair.close()

    def test_outputs_are_copies(self):
        spec = echo_spec({"x": ((4,), np.float32)}, 4)
        pair = Pair(spec)
        srv = ThreadServer(pair.server)
        try:
            out = pair.client.request({"x": np.ones((2, 4), np.float32)})
            pair.client.out_views["echo_x"][:] = 99
            self.assertTrue(np.all(out["echo_x"] == 1))
        finally:
            srv.close()
            pair.close()

    def test_sequence_numbers_advance(self):
        spec = echo_spec({"x": ((1,), np.int8)}, 2)
        pair = Pair(spec)
        srv = ThreadServer(pair.server)
        try:
            for i in range(1, 50):
                pair.client.request({"x": np.zeros((1, 1), np.int8)})
                self.assertEqual(pair.client.seq, i)
                self.assertEqual(pair.client.header[SEQ], i)
                self.assertEqual(pair.client.header[RESP_SEQ], i)
        finally:
            srv.close()
            pair.close()


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.spec = echo_spec({"a": ((3,), np.float32), "b": ((), np.int8)}, 8)
        self.pair = Pair(self.spec, timeout=0.2)

    def tearDown(self):
        self.pair.close()

    def rejected(self, inputs, error):
        before = bytes(self.pair.slots[0].buf[:self.spec.nbytes])
        with self.assertRaises(error):
            self.pair.client.request(inputs)
        self.assertEqual(bytes(self.pair.slots[0].buf[:self.spec.nbytes]), before, "a rejected request touched the slot")

    def test_rejections(self):
        a = np.zeros((4, 3), np.float32)
        b = np.zeros(4, np.int8)
        self.rejected({"a": a}, KeyError)
        self.rejected({"b": b, "a": a}, KeyError)
        self.rejected({"a": a, "b": b, "c": b}, KeyError)
        self.rejected({"a": a.astype(np.float64), "b": b}, TypeError)
        self.rejected({"a": a, "b": b.astype(np.int16)}, TypeError)
        self.rejected({"a": np.zeros((4, 4), np.float32), "b": b}, ValueError)
        self.rejected({"a": np.zeros((4, 3, 1), np.float32), "b": b}, ValueError)
        self.rejected({"a": a, "b": np.zeros((4, 1), np.int8)}, ValueError)
        self.rejected({"a": np.zeros((0, 3), np.float32), "b": np.zeros(0, np.int8)}, ValueError)
        self.rejected({"a": np.zeros((9, 3), np.float32), "b": np.zeros(9, np.int8)}, ValueError)
        self.rejected({"a": a, "b": np.zeros(3, np.int8)}, ValueError)
        self.rejected({"a": a.tolist(), "b": b}, TypeError)

    def test_outstanding_request_blocks_a_second(self):
        self.pair.client.header[STATE] = REQUEST
        with self.assertRaises(RuntimeError):
            self.pair.client.request({"a": np.zeros((1, 3), np.float32), "b": np.zeros(1, np.int8)})

    def test_timeout_without_server(self):
        with self.assertRaises(TimeoutError):
            self.pair.client.request({"a": np.zeros((1, 3), np.float32), "b": np.zeros(1, np.int8)})


class TestServerSide(unittest.TestCase):
    def test_state_rules(self):
        spec = echo_spec({"x": ((2,), np.int8)}, 4)
        pair = Pair(spec)
        try:
            server = pair.server
            self.assertEqual(server.pending(), 0)
            with self.assertRaises(RuntimeError):
                server.respond()
            server.header[STATE] = RESPONSE
            self.assertEqual(server.pending(), 0)
            server.header[STATE] = REQUEST
            server.header[COUNT] = 5
            with self.assertRaises(RuntimeError):
                server.pending()
            server.header[COUNT] = 0
            with self.assertRaises(RuntimeError):
                server.pending()
        finally:
            pair.close()

    def test_stale_response_is_detected(self):
        spec = echo_spec({"x": ((2,), np.int8)}, 4)
        pair = Pair(spec, timeout=2.0)

        def wrong_seq(server):
            serve(server)
            server.header[RESP_SEQ] = -1

        srv = ThreadServer(pair.server, mutate=wrong_seq)
        try:
            with self.assertRaises(RuntimeError):
                pair.client.request({"x": np.zeros((1, 2), np.int8)})
        finally:
            srv.close()
            pair.close()


STRESS_SPEC = echo_spec({"board": ((15, 15, 4), np.int8), "meta": ((3,), np.float32)}, 32)


def stress_client(i, shm, request_event, response_event, n_requests, results):
    rng = np.random.default_rng(1000 + i)
    client = ClientEnd(STRESS_SPEC, shm, request_event, response_event, timeout=30.0)
    bad = 0
    for _ in range(n_requests):
        n = int(rng.integers(1, 33))
        board = random_rows(rng, (15, 15, 4), np.int8, n)
        meta = random_rows(rng, (3,), np.float32, n)
        out = client.request({"board": board, "meta": meta})
        ok = (out["echo_board"].tobytes() == board.tobytes() and out["echo_meta"].tobytes() == meta.tobytes()
              and np.all(out["tag"][:, 0] == i) and np.all(out["tag"][:, 1] == client.seq)
              and np.array_equal(out["tag"][:, 2], np.arange(n)) and out["tag"].shape == (n, 3))
        bad += not ok
    client.close()
    results.put((i, bad))


def sleeper(wakeup, ready, woke):
    ready.set()
    woke.put(wakeup.wait(10.0))


class TestWakeup(unittest.TestCase):
    def test_wait_times_out_without_a_signal(self):
        self.assertFalse(Wakeup().wait(0.01))

    def test_signals_collapse_into_one(self):
        w = Wakeup()
        for _ in range(5):
            w.signal()
        self.assertTrue(w.wait(0.01))
        self.assertFalse(w.wait(0.01))

    def test_signal_wakes_a_sleeping_process_without_blocking(self):
        w, ready, woke = Wakeup(), mp.Event(), mp.Queue()
        p = mp.Process(target=sleeper, args=(w, ready, woke))
        p.start()
        try:
            ready.wait(10)
            time.sleep(0.2)
            t0 = time.perf_counter()
            w.signal()
            signal_us = 1e6 * (time.perf_counter() - t0)
            self.assertTrue(woke.get(timeout=10))
            self.assertLess(signal_us, 1000)
        finally:
            p.join(timeout=10)

    def test_pickles_into_a_child_process(self):
        ctx = mp.get_context("spawn")
        w, ready, woke = Wakeup(ctx), ctx.Event(), ctx.Queue()
        p = ctx.Process(target=sleeper, args=(w, ready, woke))
        p.start()
        try:
            ready.wait(30)
            w.signal()
            self.assertTrue(woke.get(timeout=30))
        finally:
            p.join(timeout=30)


def stress_server(slots, request_event, response_events, total, done):
    servers = [ServerEnd(STRESS_SPEC, shm, ev) for shm, ev in zip(slots, response_events)]
    served = 0
    while served < total:
        progressed = False
        for i, server in enumerate(servers):
            if serve(server, client_id=i):
                served += 1
                progressed = True
        if not progressed and request_event is not None:
            request_event.wait(0.05)
    for server in servers:
        server.close()
    done.put(served)


class TestMultiProcess(unittest.TestCase):
    def run_stress(self, use_events, clients=6, n_requests=300):
        slots = create_slots(STRESS_SPEC, clients)
        request_event = Wakeup() if use_events else None
        response_events = [Wakeup() if use_events else None for _ in range(clients)]
        results, done = mp.Queue(), mp.Queue()
        server = mp.Process(target=stress_server,
                            args=(slots, request_event, response_events, clients * n_requests, done))
        procs = [mp.Process(target=stress_client, args=(i, slots[i], request_event, response_events[i],
                                                         n_requests, results)) for i in range(clients)]
        try:
            server.start()
            for p in procs:
                p.start()
            outcome = dict(results.get(timeout=120) for _ in procs)
            self.assertEqual(done.get(timeout=60), clients * n_requests)
            for p in procs + [server]:
                p.join(timeout=30)
                self.assertEqual(p.exitcode, 0)
            self.assertEqual(outcome, {i: 0 for i in range(clients)})
        finally:
            for p in procs + [server]:
                if p.is_alive():
                    p.terminate()
            release_slots(slots)

    def test_stress_with_events(self):
        self.run_stress(use_events=True)

    def test_stress_spinning(self):
        self.run_stress(use_events=False, clients=4, n_requests=200)


if __name__ == "__main__":
    unittest.main()
