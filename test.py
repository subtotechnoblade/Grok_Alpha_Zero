# feel free to comment everything out and test your own code if you'd like
# Brian's test section
import sys
import time
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _get_best_PUCT_score_index(child_prob_priors: np.array,
                               child_values: np.array,
                               child_visits: np.array,
                               parent_visits: int,
                               c_puct_init: float = 2.5,
                               c_puct_base: float = 19652):
    # note that np.log is actually a math ln with base e (2.7)
    U = child_prob_priors * ((parent_visits** 0.5) / (child_visits + 1)) * (
                c_puct_init + np.log((parent_visits + c_puct_base + 1) / c_puct_base))
    PUCT_score = (child_values / child_visits) + U
    return np.argmax(PUCT_score)


if __name__ == "__main__":
    prior = np.random.uniform(0, 1, 255)
    values = np.random.uniform(0, 100, 255)
    visits = np.random.randint(1, 200, 255)
    parent_visits = np.sum(visits)

    s = time.time()
    for _ in range(1000):
        _get_best_PUCT_score_index(prior, values, visits, parent_visits)
    print(time.time() - s)
    print(_get_best_PUCT_score_index(prior, values, visits, parent_visits))





