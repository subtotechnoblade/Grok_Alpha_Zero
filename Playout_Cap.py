"""Playout cap randomization with escalation.

Each move is either a full search (recorded as training data) or a cheap one that only picks the move. A cheap
search escalates to a full one, continuing the same tree, when the network looks wrong about the position:
    policy: the search chose a move the network rated below its favourite by more than escalate_disagreement
            (log of the prior ratio)
    value:  |root Q - network value| is above escalate_value_gap
Both thresholds are absolute, so escalation is common while the network is weak and fades as it improves.
"""
import numpy as np

REASONS = ("explore", "random", "policy", "value", "close")


def prior_disagreement(child_visits, child_priors):
    """log(prior of the network's favourite / prior of the most visited move): 0 when search and network agree,
    log 2 when the search picked a move the network rated at half its favourite. Independent of prior flatness."""
    priors = np.maximum(np.asarray(child_priors, dtype=np.float64), 1e-12)
    visits = np.asarray(child_visits)
    if visits.sum() <= 0:
        return 0.0
    return float(np.log(priors.max() / priors[np.argmax(visits)]))


def value_gap(child_visits, child_values, net_value):
    """|root Q - network value|, both from the player to move."""
    visits = np.asarray(child_visits, dtype=np.float64)
    total = visits.sum()
    if total <= 0:
        return 0.0
    return abs(float(np.asarray(child_values, dtype=np.float64).sum() / total) - float(net_value))


def policy_surprise(child_visits, child_priors):
    """KL(search visits || prior), the policy-surprise weight signal."""
    visits = np.asarray(child_visits, dtype=np.float64)
    total = visits.sum()
    if total <= 0:
        return 0.0
    priors = np.maximum(np.asarray(child_priors, dtype=np.float64), 1e-8)
    pi = visits / total
    seen = pi > 0
    return float(np.sum(pi[seen] * np.log(pi[seen] / priors[seen])))


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
    def __init__(self, train_config):
        self.full_search_prob = float(train_config.get("full_search_prob", 1.0))
        self.cheap_sims = int(train_config.get("cheap_sims", 128))
        self.triggers = {name: bool(train_config.get(f"escalate_{name}", False)) for name in ("policy", "value")}
        self.disagreement = float(train_config.get("escalate_disagreement", np.log(2)))
        self.value_gap = float(train_config.get("escalate_value_gap", 0.3))

    @property
    def enabled(self):
        return self.full_search_prob < 1.0

    @property
    def needs_net_value(self):
        return self.triggers["value"]

    def full_search(self):
        return np.random.random() < self.full_search_prob

    def escalate(self, child_visits, child_values, child_priors, net_value=0.0):
        """The reason a cheap search should become a full one, or None."""
        if self.triggers["policy"] and prior_disagreement(child_visits, child_priors) > self.disagreement:
            return "policy"
        if self.triggers["value"] and value_gap(child_visits, child_values, net_value) > self.value_gap:
            return "value"
        return None
