"""Quoridor rules, canonical frame and mirror symmetry. python -m unittest tests.test_quoridor"""
import unittest

import numpy as np

from MCTS import MCTS
from Quoridor.game import Quoridor, SIZE, WALLS, MIRROR_PERM, FLIP_PERM, mirror_states, _policy_index
from Session_Cache import Cache_Wrapper
from tests.fake_net import FakeSession

N = SIZE


def blocked(board, r, c, nr, nc):
    H, V = board[0], board[1]
    if nc == c:
        top = min(r, nr)
        return any(H[top, gc] for gc in (c - 1, c) if 0 <= gc < N - 1)
    left = min(c, nc)
    return any(V[gr, left] for gr in (r - 1, r) if 0 <= gr < N - 1)


def neighbours(board, r, c):
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < N and 0 <= nc < N and not blocked(board, r, c, nr, nc):
            yield nr, nc


def reachable(board, start, goal_row):
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        if r == goal_row:
            return True
        for nxt in neighbours(board, r, c):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def reference_actions(board, player):
    """Brute force: every pawn rule spelled out, every wall checked with a full search for both players."""
    me = tuple(int(v) for v in board[2, 1 if player == -1 else 2, :2])
    them = tuple(int(v) for v in board[2, 2 if player == -1 else 1, :2])
    actions = set()
    for nxt in neighbours(board, *me):
        if nxt != them:
            actions.add((0, *nxt))
            continue
        dr, dc = nxt[0] - me[0], nxt[1] - me[1]
        behind = (nxt[0] + dr, nxt[1] + dc)
        if behind in set(neighbours(board, *nxt)):
            actions.add((0, *behind))
        else:
            for side in neighbours(board, *nxt):
                if side != me and (side[0] - nxt[0], side[1] - nxt[1]) != (dr, dc):
                    actions.add((0, *side))
    if board[2, 0, 0 if player == -1 else 1] == 0:
        return actions
    black, white = tuple(int(v) for v in board[2, 1, :2]), tuple(int(v) for v in board[2, 2, :2])
    H, V = board[0], board[1]
    for gr in range(N - 1):
        for gc in range(N - 1):
            for t, plane in ((1, H), (2, V)):
                if H[gr, gc] or V[gr, gc]:
                    continue
                if t == 1 and any(H[gr, x] for x in (gc - 1, gc + 1) if 0 <= x < N - 1):
                    continue
                if t == 2 and any(V[x, gc] for x in (gr - 1, gr + 1) if 0 <= x < N - 1):
                    continue
                trial = board.copy()
                trial[t - 1, gr, gc] = 1
                if reachable(trial, black, 0) and reachable(trial, white, N - 1):
                    actions.add((t, gr, gc))
    return actions


def random_game(rng, max_plies=60, wall_bias=0.4):
    game = Quoridor()
    positions = [(game.board.copy(), game.next_player)]
    for _ in range(max_plies):
        legal = game.get_legal_actions()
        walls = legal[legal[:, 0] > 0]
        pool = walls if len(walls) and rng.random() < wall_bias else legal[legal[:, 0] == 0]
        game.do_action(pool[rng.integers(len(pool))])
        if game.check_win() != -2:
            break
        positions.append((game.board.copy(), game.next_player))
    return game, positions


def mirror_board(board):
    out = board.copy()
    out[0, :N - 1, :N - 1] = board[0, :N - 1, N - 2::-1]
    out[1, :N - 1, :N - 1] = board[1, :N - 1, N - 2::-1]
    out[2, 1, 1] = N - 1 - board[2, 1, 1]
    out[2, 2, 1] = N - 1 - board[2, 2, 1]
    return out


def flip_board(board):
    """Swaps colours and turns the board upside down, so black's position becomes white's."""
    out = board.copy()
    out[0, :N - 1, :N - 1] = board[0, N - 2::-1, :N - 1]
    out[1, :N - 1, :N - 1] = board[1, N - 2::-1, :N - 1]
    out[2, 0, 0], out[2, 0, 1] = board[2, 0, 1], board[2, 0, 0]
    out[2, 1, 0], out[2, 1, 1] = N - 1 - board[2, 2, 0], board[2, 2, 1]
    out[2, 2, 0], out[2, 2, 1] = N - 1 - board[2, 1, 0], board[2, 1, 1]
    return out


def legal(board, player):
    return Quoridor.get_legal_actions_MCTS(board, -player, np.zeros((0, 3), dtype=np.uint8))


def state(board, player):
    return Quoridor.get_input_state_MCTS(board, -player, np.zeros((0, 3), dtype=np.uint8))


class TestRules(unittest.TestCase):
    def test_start(self):
        game = Quoridor()
        self.assertEqual(game.pawn(-1), (N - 1, N // 2))
        self.assertEqual(game.pawn(1), (0, N // 2))
        self.assertEqual(game.walls_left(-1), WALLS)
        self.assertEqual(len(game.get_legal_actions()), 3 + 2 * (N - 1) ** 2)

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(60):
            _, positions = random_game(rng)
            for board, player in positions:
                fast = {tuple(int(v) for v in a) for a in legal(board, player)}
                self.assertEqual(len(fast), len(legal(board, player)))
                self.assertEqual(fast, reference_actions(board, player))
                checked += 1
        self.assertGreater(checked, 500)

    def test_jumps(self):
        game = Quoridor()
        game.board[2, 1, :2] = (2, 2)
        game.board[2, 2, :2] = (1, 2)
        moves = {tuple(a[1:]) for a in game.get_legal_actions() if a[0] == 0}
        self.assertEqual(moves, {(0, 2), (3, 2), (2, 1), (2, 3)})
        game.board[0, 0, 1] = 1
        moves = {tuple(a[1:]) for a in game.get_legal_actions() if a[0] == 0}
        self.assertEqual(moves, {(1, 1), (1, 3), (3, 2), (2, 1), (2, 3)})

    def test_wall_spends_and_win(self):
        game = Quoridor()
        game.do_action((1, 2, 2))
        self.assertEqual(game.walls_left(-1), WALLS - 1)
        self.assertEqual(game.board[0, 2, 2], 1)
        game.board[2, 1, :2] = (1, 0)
        game.next_player = -1
        self.assertEqual(game.check_win(), -2)
        game.do_action((0, 0, 0))
        self.assertEqual(game.check_win(), -1)

    def test_no_walls_left(self):
        game = Quoridor()
        game.board[2, 0, 0] = 0
        self.assertTrue(all(a[0] == 0 for a in game.get_legal_actions()))


class TestAuxTargets(unittest.TestCase):
    def test_values_from_final_board(self):
        game = Quoridor()
        game.do_action((1, 2, 2))
        game.do_action((0, 1, N // 2))
        game.do_action((0, N - 2, N // 2))
        aux = Quoridor.compute_aux_targets(game.board, np.array(game.action_history))
        self.assertEqual(aux.shape, (3, Quoridor.num_aux_heads, 1))
        self.assertEqual(aux.dtype, np.float32)
        black_dist, white_dist = N - 2, N - 2
        np.testing.assert_allclose(aux[:, 0, 0], [0.03, 0.02, 0.01], rtol=1e-6)
        np.testing.assert_allclose(aux[0, :, 0], [0.03, black_dist / N, white_dist / N, (WALLS - 1) / WALLS, 1.0], rtol=1e-6)
        np.testing.assert_allclose(aux[1, 1:, 0], [white_dist / N, black_dist / N, 1.0, (WALLS - 1) / WALLS], rtol=1e-6)
        np.testing.assert_array_equal(aux[2], aux[0] - np.array([[0.02], [0], [0], [0], [0]], dtype=np.float32))

    def test_winner_ends_at_zero(self):
        game, _ = random_game(np.random.default_rng(5), max_plies=400, wall_bias=0.1)
        winner = game.check_win()
        aux = Quoridor.compute_aux_targets(game.board, np.array(game.action_history))
        self.assertEqual(len(aux), len(game.action_history))
        if winner != -2:
            self.assertEqual(aux[-1, 1, 0], 0.0)
            self.assertGreater(aux[-1, 2, 0], 0.0)


class TestFrames(unittest.TestCase):
    def positions(self, seed=1):
        rng = np.random.default_rng(seed)
        for _ in range(20):
            for item in random_game(rng)[1]:
                yield item

    def test_mirror_state_matches_mirrored_board(self):
        for board, player in self.positions():
            np.testing.assert_array_equal(mirror_states(state(board, player)), state(mirror_board(board), player))

    def test_white_sees_what_black_sees(self):
        for board, player in self.positions(2):
            np.testing.assert_array_equal(state(board, player)[..., :13], state(flip_board(board), -player)[..., :13])

    def test_policy_indices(self):
        for board, player in self.positions(3):
            actions = legal(board, player)
            idx = [_policy_index(int(t), int(r), int(c), N, player == 1) for t, r, c in actions]
            self.assertEqual(len(set(idx)), len(idx))
            mirrored = {_policy_index(int(t), int(r), int(c), N, player == 1) for t, r, c in legal(mirror_board(board), player)}
            self.assertEqual({int(MIRROR_PERM[i]) for i in idx}, mirrored)
            flipped = {_policy_index(int(t), int(r), int(c), N, player == -1) for t, r, c in legal(flip_board(board), -player)}
            self.assertEqual(set(idx), flipped)

    def test_policy_target_round_trip(self):
        game = Quoridor()
        game.do_action((0, N - 2, N // 2))
        actions = game.get_legal_actions()
        probs = np.random.default_rng(0).random(len(actions))
        probs /= probs.sum()
        target = game.compute_policy_improvement(zip(actions, probs))
        _, prior = Quoridor.get_legal_actions_policy_MCTS(game.board, -1, np.zeros((0, 3), np.uint8), target.copy())
        np.testing.assert_allclose(prior, probs, rtol=1e-5)

    def test_opponent_reply_lands_in_movers_frame(self):
        for board, player in self.positions(5):
            for t, r, c in legal(board, -player):
                theirs = np.zeros(3 * N * N, np.float32)
                theirs[_policy_index(int(t), int(r), int(c), N, -player == 1)] = 1.0
                mine = np.zeros(3 * N * N, np.float32)
                mine[_policy_index(int(t), int(r), int(c), N, player == 1)] = 1.0
                np.testing.assert_array_equal(theirs[FLIP_PERM], mine)
        np.testing.assert_array_equal(FLIP_PERM[FLIP_PERM], np.arange(3 * N * N))
        np.testing.assert_array_equal(FLIP_PERM[MIRROR_PERM], MIRROR_PERM[FLIP_PERM])

    def test_augment_and_cache_symmetry(self):
        boards = [state(b, p) for b, p in self.positions(4)][:30]
        states = np.array(boards)
        policies = np.random.default_rng(0).random((len(states), 3 * N * N)).astype(np.float32)
        aug_states, aug_policies = Quoridor().augment_sample(states, policies)
        self.assertEqual(aug_states.shape, (2, *states.shape))
        self.assertEqual(aug_states.dtype, np.int8)
        np.testing.assert_array_equal(aug_states[1], mirror_states(states))
        ids = np.arange(len(states)) % 2
        moved = Quoridor.apply_symmetry(states, ids)
        np.testing.assert_array_equal(Quoridor.apply_symmetry(moved, ids), states)
        np.testing.assert_array_equal(Quoridor.undo_symmetry(Quoridor.undo_symmetry(policies, ids), ids), policies)


class TestSearch(unittest.TestCase):
    def test_search_plays_a_game(self):
        game = Quoridor()
        session = Cache_Wrapper(FakeSession(game.get_input_state().shape, 3 * N * N, scale=0.02), 2,
                                symmetries=Quoridor, random_symmetries=True)
        for _ in range(40):
            mcts = MCTS(game, session, use_dirichlet=False, tau=0.0, fpu_reduction=0.2, leaves_per_batch=4)
            action, probs = mcts.run(iteration_limit=200, use_bar=False)
            self.assertTrue(any(np.array_equal(action, a) for a in game.get_legal_actions()))
            self.assertAlmostEqual(sum(p[1] for p in probs), 1.0, places=5)
            game.do_action(action)
            if game.check_win() != -2:
                break

    def test_takes_the_winning_step(self):
        game = Quoridor()
        game.board[2, 1, :2] = (1, 0)
        mcts = MCTS(game, None, use_dirichlet=False, tau=0.0)
        action, _ = mcts.run(iteration_limit=100, use_bar=False)
        self.assertEqual(tuple(int(v) for v in action), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
