"""Playout cap randomization with escalation.

Each move is either a full search (recorded as training data) or a cheap one that only picks the move. A cheap
search escalates to a full one, continuing the same tree, when it looks surprising:
    policy: KL(cheap visits || prior) above a running percentile
    value:  |root Q - network value| above a running percentile
    close:  the Q gap between the two best moves (with enough visits) below the matching running percentile,
            and below close_gap
Running percentiles are over this worker's recent cheap searches, so each trigger fires on about
(100 - escalate_percentile)% of them however the network improves.
"""
from collections import deque

import numpy as np

REASONS = ("explore", "random", "policy", "value", "close")


def root_signals(child_visits, child_values, child_priors, net_value, close_min_visits):
    visits = np.asarray(child_visits, dtype=np.float64)
    total = visits.sum()
    if total <= 0:
        return 0.0, False, 0.0, np.inf
    priors = np.maximum(np.asarray(child_priors, dtype=np.float64), 1e-8)
    pi = visits / total
    seen = pi > 0
    kl = float(np.sum(pi[seen] * np.log(pi[seen] / priors[seen])))
    argmax_changed = bool(np.argmax(visits) != np.argmax(priors))
    values = np.asarray(child_values, dtype=np.float64)
    value_surprise = abs(float(values.sum() / total) - float(net_value))
    q = values[visits >= close_min_visits] / visits[visits >= close_min_visits]
    gap = float(np.diff(np.sort(q)[-2:])[0]) if len(q) >= 2 else np.inf
    return kl, argmax_changed, value_surprise, gap


def completed_q_policy(child_visits, child_values, child_priors, net_value, c_visit=50.0, c_scale=0.1):
    """Gumbel MuZero's improved policy softmax(log prior + sigma(completed Q)), from any search tree's root stats.
    Unvisited moves get v_mix, the prior-weighted mean of the visited Q values mixed with the network value."""
    visits = np.asarray(child_visits, dtype=np.float64)
    values = np.asarray(child_values, dtype=np.float64)
    priors = np.maximum(np.asarray(child_priors, dtype=np.float64), 1e-12)
    priors /= priors.sum()
    seen = visits > 0
    total = visits.sum()
    q = np.where(seen, values / np.maximum(visits, 1), 0.0)
    weighted_q = (priors[seen] * q[seen]).sum() / priors[seen].sum() if seen.any() else 0.0
    v_mix = (float(net_value) + weighted_q * total) / (total + 1)
    completed = np.where(seen, q, v_mix)
    span = completed.max() - completed.min()
    completed = (completed - completed.min()) / span if span > 0 else np.zeros_like(completed)
    logits = np.log(priors) + (c_visit + visits.max()) * c_scale * completed
    logits -= logits.max()
    with np.errstate(under="ignore"):
        policy = np.exp(logits)
        return (policy / policy.sum()).astype(np.float32)


class Playout_Cap:
    def __init__(self, train_config, history=1000, min_history=50):
        self.full_search_prob = float(train_config.get("full_search_prob", 1.0))
        self.cheap_sims = int(train_config.get("cheap_sims", 128))
        self.triggers = {name: bool(train_config.get(f"escalate_{name}", False)) for name in ("policy", "value", "close")}
        self.percentile = float(train_config.get("escalate_percentile", 90.0))
        self.close_gap = float(train_config.get("close_gap", 0.02))
        self.close_min_visits = int(train_config.get("close_min_visits", 10))
        self.min_history = min_history
        self.history = {name: deque(maxlen=history) for name in ("policy", "value", "close")}

    @property
    def enabled(self):
        return self.full_search_prob < 1.0

    @property
    def needs_net_value(self):
        return self.triggers["value"]

    def full_search(self):
        return np.random.random() < self.full_search_prob

    def _above(self, name, signal):
        history = self.history[name]
        above = len(history) >= self.min_history and signal > np.percentile(history, self.percentile)
        history.append(signal)
        return above

    def escalate(self, child_visits, child_values, child_priors, net_value=0.0):
        """The reason a cheap search should become a full one, or None."""
        kl, _, value_surprise, gap = root_signals(child_visits, child_values, child_priors, net_value,
                                                  self.close_min_visits)
        reason = None
        if self.triggers["policy"] and self._above("policy", kl):
            reason = "policy"
        if self.triggers["value"] and self._above("value", value_surprise) and reason is None:
            reason = "value"
        if self.triggers["close"] and np.isfinite(gap) and self._above("close", -gap) and gap < self.close_gap \
                and reason is None:
            reason = "close"
        return reason
