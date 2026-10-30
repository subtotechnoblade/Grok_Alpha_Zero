import numpy as np


class Cache_Wrapper:
    """In-memory memo of neural network outputs for the opening region of the game.

    Wraps any object with a .run(output_names, input_feed) method (an onnxruntime
    InferenceSession or a Parallelized_Session) and memoises its outputs.

    Lifetime is the worker PROCESS, which is one generation. That is deliberate: the
    memo maps position -> (policy, value) as judged by ONE network, so every entry
    becomes a stale opinion the moment the network retrains. Dying with the process IS
    the invalidation mechanism - there is nothing to invalidate by hand, and nothing is
    persisted to disk. See tmp/REWRITE_PLAN.md section 6b.

    Only positions at depth < max_cache_depth are stored, because only the opening
    region repeats across games. Deep positions are visited once and never again, so
    hashing them would be pure overhead.

    NOTE: MCTS.py type-checks `isinstance(session, Cache_Wrapper)` to decide whether to
    pass `depth`, so this class name and the run() signature are load bearing.
    """

    def __init__(self, session, max_cache_depth=2, max_entries=100_000, symmetries=None, average_symmetries=True,
                 random_symmetries=False, max_rows=None):
        self.session = session
        self.max_cache_depth = max_cache_depth
        self.max_entries = max_entries  # safety bound only, there is no eviction:
        # opening positions repeat and deep ones are never stored, so LRU buys nothing
        self.symmetries = symmetries  # game class with apply_symmetry / undo_symmetry / num_symmetries
        self.average_symmetries = average_symmetries  # cacheable rows: mean over every symmetry
        self.random_symmetries = random_symmetries  # other rows: one random symmetry each
        self.max_rows = max_rows  # rows per session call, the inference server's slot size

        self.cache = {}
        self.hits = 0
        self.misses = 0

    def run(self, output_names, input_feed: dict, depth):
        return self.run_batch(output_names, input_feed["inputs"], (depth,))

    def run_batch(self, output_names, inputs, depths):
        """Rows are looked up independently: hits come from the memo, misses go to the session as one
        sub-batch, and outputs come back in row order. Stores and returns per-row outputs."""
        rows = None
        keys = {}
        for i, depth in enumerate(depths):
            if depth < self.max_cache_depth:
                # too deep to ever be seen twice, don't pay for the hash
                key = inputs[i].tobytes()
                hit = self.cache.get(key)
                if hit is not None:
                    self.hits += 1
                    if rows is None:
                        rows = [None] * len(inputs)
                    rows[i] = hit
                    # safe to hand back the same arrays: get_legal_actions_policy_MCTS fancy
                    # indexes the policy (policy[legal_actions]) which copies, and every later
                    # in place op is on that copy. No caller mutates what we return.
                    continue
                self.misses += 1
                keys[i] = key

        if rows is None:
            outputs = self._evaluate(output_names, inputs, [i in keys for i in range(len(inputs))])
            for i, key in keys.items():
                if len(self.cache) < self.max_entries:
                    self.cache[key] = [output[i].copy() for output in outputs]
            return outputs

        miss = [i for i in range(len(inputs)) if rows[i] is None]
        if miss:
            outputs = self._evaluate(output_names, inputs[miss], [i in keys for i in miss])
            for j, i in enumerate(miss):
                rows[i] = [output[j].copy() for output in outputs]
                if i in keys and len(self.cache) < self.max_entries:
                    self.cache[keys[i]] = rows[i]
        return [np.stack([row[k] for row in rows]) for k in range(len(output_names))]

    def _session_run(self, output_names, inputs):
        if self.max_rows is None or len(inputs) <= self.max_rows:
            return self.session.run(output_names, {"inputs": inputs})
        parts = [self.session.run(output_names, {"inputs": inputs[i:i + self.max_rows]})
                 for i in range(0, len(inputs), self.max_rows)]
        return [np.concatenate([part[k] for part in parts]) for k in range(len(output_names))]

    def _evaluate(self, output_names, inputs, cacheable):
        average = np.asarray(cacheable, dtype=bool) & self.average_symmetries
        if self.symmetries is None or not (average.any() or self.random_symmetries):
            return self._session_run(output_names, inputs)
        game, n, m = self.symmetries, self.symmetries.num_symmetries, int(average.sum())
        single = inputs[~average]
        single_ids = (np.random.randint(n, size=len(single)) if self.random_symmetries
                      else np.zeros(len(single), dtype=np.int64))
        ids = np.concatenate([np.repeat(np.arange(n), m), single_ids])
        states = np.concatenate([np.tile(inputs[average], (n, *[1] * (inputs.ndim - 1))), single])
        raw = dict(zip(output_names, self._session_run(output_names, game.apply_symmetry(states, ids))))
        raw["policy"] = game.undo_symmetry(raw["policy"], ids)
        outputs = []
        for name in output_names:
            out = np.empty((len(inputs), *raw[name].shape[1:]), dtype=raw[name].dtype)
            out[average] = raw[name][:n * m].reshape(n, m, *raw[name].shape[1:]).mean(0)
            out[~average] = raw[name][n * m:]
            outputs.append(out)
        return outputs

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
                "entries": len(self.cache)}
