import numpy as np
from diskcache import Cache

class Cache_Wrapper:
    def __init__(self, session, path, max_cache_actions=20):
        self.session = session
        self.path = path

        self.finished_lookup = False if max_cache_actions > 0 else 0
        self.cache = Cache(path, timeout=100000)
        self.max_cache_actions = max_cache_actions
        self.action_count = 0

    def run(self, output_names, input_feed:dict):
        if not self.finished_lookup:
            key = np.ascontiguousarray(input_feed["inputs"].flatten()).newbyteorder("little").tobytes()
            outputs, expire_time = self.cache.get(key, expire_time=True)
            if outputs is not None:
                self.action_count += 1
                return outputs

            self.finished_lookup = True
        outputs = self.session.run(output_names, input_feed)

        if self.action_count < self.max_cache_actions:
            key = np.ascontiguousarray(input_feed["inputs"].flatten()).newbyteorder("little").tobytes()
            self.cache.set(key, outputs)
            self.action_count += 1
        return outputs
    def __del__(self):
        self.cache.close()


if __name__ == "__main__":
    import time

    x = np.ones((225,)) * 1
    iterations = 10000
    s = time.time()
    for _ in range(iterations):
        hash(tuple(x.flatten()))
    print(iterations / (time.time() - s))
    # print(time.time() - s)
    # print(time.time() - s)

