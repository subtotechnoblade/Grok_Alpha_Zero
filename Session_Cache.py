import time
import numpy as np
from diskcache import Cache

class Cache_Wrapper:
    def __init__(self, session, path, save_moves=10):
        self.session = session
        self.path = path

        self.finished_lookup = False
        self.cache = Cache(path, timeout=100000)
        self.save_moves = save_moves
        self.move_count = 0

    def run(self, output_names, input_feed:dict):
        key = hash(tuple(input_feed["inputs"].flatten()))
        if not self.finished_lookup:
            outputs, expire_time = self.cache.get(key, expire_time=True)
            if outputs is not None:
                if self.move_count > self.save_moves:
                    self.cache.touch(key, time.time() + 2 * 60)
                self.move_count += 1
                return outputs

            self.finished_lookup = True
        outputs = self.session.run(output_names, input_feed)

        if self.move_count <= self.save_moves:
            expire = None
        else:
            expire = 5 * 60.0
        self.cache.set(key, outputs, expire=expire, retry=True)
        self.move_count += 1
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

