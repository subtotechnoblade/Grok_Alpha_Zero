import numpy as np
from diskcache import Cache

class Cache_Wrapper:
    def __init__(self, session, path, max_cache_depth=2):
        self.session = session
        self.path = path

        self.finished_lookup = False if max_cache_depth > 0 else 0
        self.cache = Cache(path,
                           timeout=100000
                           )
        self.max_cache_depth = max_cache_depth

    def run(self, output_names, input_feed:dict, depth):
        if not self.finished_lookup and self.max_cache_depth > 0:
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

