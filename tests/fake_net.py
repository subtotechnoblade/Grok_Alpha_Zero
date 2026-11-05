import numpy as np


class FakeSession:
    """Deterministic stand-in for an onnx session. Each row's outputs depend only on that row,
    so any batching of the same inputs returns bit-identical rows."""

    def __init__(self, input_shape, policy_size, seed=0, scale=0.3):
        rng = np.random.default_rng(seed)
        in_size = int(np.prod(input_shape))
        self.w_policy = rng.standard_normal((in_size, policy_size)) * scale
        self.w_value = rng.standard_normal(in_size) * 0.05
        self.w_draw = rng.standard_normal(in_size) * 0.05
        self.calls = 0
        self.rows = 0

    def run(self, output_names, input_feed, run_options=None):
        x = np.asarray(input_feed["inputs"], dtype=np.float64)
        x = x.reshape(x.shape[0], -1)
        policy = np.empty((x.shape[0], self.w_policy.shape[1]), dtype=np.float32)
        value = np.empty((x.shape[0], 1), dtype=np.float32)
        draw = np.empty((x.shape[0], 1), dtype=np.float32)
        for i in range(x.shape[0]):
            logits = x[i] @ self.w_policy
            e = np.exp(logits - logits.max())
            policy[i] = e / e.sum()
            value[i, 0] = np.tanh(x[i] @ self.w_value)
            draw[i, 0] = (1.0 - abs(value[i, 0])) / (1.0 + np.exp(-(x[i] @ self.w_draw)))
        self.calls += 1
        self.rows += x.shape[0]
        outputs = {"policy": policy, "value": value, "draw": draw}
        return [outputs[name] for name in output_names]
