"""Play Quoridor against the latest network.
python Quoridor/play.py [--human black|white|both|none] [--gen N] [--sims 800] [--cpu]
Click a highlighted square to move. Hover a groove between squares to preview a wall, click to place it.
U undo, N new game, Esc quit."""
import argparse
import os
import sys
import threading
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as rt
import pygame

from MCTS import MCTS
from Quoridor.game import Quoridor, SIZE

N = SIZE
CELL = 600 // N
GAP = max(6, CELL // 7)
MARGIN = 30
PANEL = 260
BOARD_PX = N * CELL
WIDTH, HEIGHT = MARGIN * 2 + BOARD_PX + PANEL, MARGIN * 2 + BOARD_PX

BG = (40, 30, 24)
SQUARE = (190, 150, 105)
GROOVE = (70, 50, 38)
HIGHLIGHT = (120, 200, 120)
WALL = (235, 225, 200)
PREVIEW = (255, 240, 150)
TEXT = (240, 235, 225)
PIECE = {-1: (25, 25, 25), 1: (245, 245, 245)}
NAME = {-1: "Black", 1: "White"}


def latest_model(gen):
    run_dir = ROOT / "Quoridor" / "Grok_Zero_Train"
    if gen is None:
        gens = sorted(int(Path(p).parent.name) for p in glob(str(run_dir / "*" / "model.onnx")))
        if not gens:
            return None, None
        gen = gens[-1]
    path = run_dir / str(gen) / "model.onnx"
    return (str(path), gen) if path.exists() else (None, None)


def make_session(path, cpu):
    if path is None:
        return None
    providers = ["CPUExecutionProvider"] if cpu else [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                                                      if p in rt.get_available_providers()]
    return rt.InferenceSession(path, providers=providers)


def square_rect(r, c):
    return pygame.Rect(MARGIN + c * CELL + GAP // 2, MARGIN + r * CELL + GAP // 2, CELL - GAP, CELL - GAP)


def wall_rect(t, gr, gc):
    x0, y0 = MARGIN + gc * CELL, MARGIN + gr * CELL
    if t == 1:
        return pygame.Rect(x0 + GAP // 2, y0 + CELL - GAP // 2, 2 * CELL - GAP, GAP)
    return pygame.Rect(x0 + CELL - GAP // 2, y0 + GAP // 2, GAP, 2 * CELL - GAP)


def hovered_wall(pos):
    """Nearest groove under the mouse as (type, groove row, groove col), or None when over a square's middle."""
    x, y = (pos[0] - MARGIN) / CELL, (pos[1] - MARGIN) / CELL
    if not (0 <= x <= N and 0 <= y <= N):
        return None
    dy, dx = abs(y - round(y)), abs(x - round(x))
    edge = GAP / CELL
    if min(dx, dy) > edge:
        return None
    if dy <= dx:
        gr = int(round(y)) - 1
        gc = min(max(int(x - 0.5), 0), N - 2)
        t = 1
    else:
        gc = int(round(x)) - 1
        gr = min(max(int(y - 0.5), 0), N - 2)
        t = 2
    if not (0 <= gr <= N - 2 and 0 <= gc <= N - 2):
        return None
    return t, gr, gc


def hovered_square(pos):
    c, r = (pos[0] - MARGIN) // CELL, (pos[1] - MARGIN) // CELL
    return (int(r), int(c)) if 0 <= r < N and 0 <= c < N else None


class AI:
    def __init__(self, session, sims):
        self.session, self.sims = session, sims
        self.thread = None
        self.result = None
        self.eval = None

    def start(self, game):
        snapshot = Quoridor()
        for action in game.action_history:
            snapshot.do_action(action)
        self.result = None
        self.thread = threading.Thread(target=self._think, args=(snapshot,), daemon=True)
        self.thread.start()

    def _think(self, game):
        mcts = MCTS(game, self.session, use_dirichlet=False, tau=0.0, c_puct_init=2.0, fpu_reduction=0.2,
                    leaves_per_batch=8, fast_find_win=True)
        action, probs = mcts.run(iteration_limit=self.sims, use_bar=False)
        best = next(p for p in probs if np.array_equal(p[0], action))
        self.eval = float(best[2])
        self.result = np.asarray(action, dtype=np.uint8)

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()


def draw(screen, font, small, game, humans, legal, hover, ai, info):
    screen.fill(BG)
    pygame.draw.rect(screen, GROOVE, (MARGIN, MARGIN, BOARD_PX, BOARD_PX))
    human_turn = game.next_player in humans and game.check_win() == -2 and not ai.busy
    targets = {(int(a[1]), int(a[2])) for a in legal if a[0] == 0} if human_turn else set()
    for r in range(N):
        for c in range(N):
            colour = HIGHLIGHT if (r, c) in targets else SQUARE
            pygame.draw.rect(screen, colour, square_rect(r, c), border_radius=4)
    for r in (0, N - 1):
        label = small.render("White goal" if r == N - 1 else "Black goal", True, TEXT)
        screen.blit(label, (MARGIN, 6 if r == 0 else HEIGHT - 22))
    for t in (1, 2):
        for gr, gc in zip(*np.nonzero(game.board[t - 1])):
            pygame.draw.rect(screen, WALL, wall_rect(t, gr, gc), border_radius=3)
    if human_turn and hover is not None and game.walls_left(game.next_player) > 0:
        legal_walls = {tuple(int(v) for v in a) for a in legal if a[0] > 0}
        colour = PREVIEW if hover in legal_walls else (200, 80, 80)
        pygame.draw.rect(screen, colour, wall_rect(*hover), border_radius=3)
    if game.action_history:
        last = game.action_history[-1]
        if last[0] == 0:
            pygame.draw.rect(screen, (220, 90, 60), square_rect(int(last[1]), int(last[2])), 3, border_radius=4)
        else:
            pygame.draw.rect(screen, (220, 90, 60), wall_rect(int(last[0]), int(last[1]), int(last[2])), 2, border_radius=3)
    for player in (-1, 1):
        r, c = game.pawn(player)
        rect = square_rect(r, c)
        pygame.draw.circle(screen, PIECE[player], rect.center, rect.width * 0.36)
        pygame.draw.circle(screen, (120, 90, 60), rect.center, rect.width * 0.36, 2)

    x = MARGIN * 2 + BOARD_PX
    lines = []
    winner = game.check_win()
    if winner != -2:
        lines.append((f"{NAME[winner]} wins!", font))
    elif ai.busy:
        lines.append((f"{NAME[game.next_player]} (AI) thinking...", font))
    else:
        lines.append((f"{NAME[game.next_player]} to move", font))
    lines.append(("", small))
    for player in (-1, 1):
        who = "you" if player in humans else "AI"
        lines.append((f"{NAME[player]} ({who}): {game.walls_left(player)} walls", small))
    lines.append(("", small))
    if ai.eval is not None:
        lines.append((f"AI eval of its move: {ai.eval:+.2f}", small))
    lines += [(line, small) for line in info]
    lines += [("", small), ("U undo   N new game", small), ("Esc quit", small)]
    y = MARGIN
    for text, f in lines:
        screen.blit(f.render(text, True, TEXT), (x, y))
        y += f.get_linesize() + 2
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="black", choices=["black", "white", "both", "none"])
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    humans = {"black": {-1}, "white": {1}, "both": {-1, 1}, "none": set()}[args.human]

    path, gen = latest_model(args.gen)
    session = make_session(path, args.cpu)
    providers = session.get_providers()[0] if session is not None else "none"
    info = [f"Network: gen {gen}" if path else "Network: none (pure search)", f"Provider: {providers}",
            f"Sims per move: {args.sims}", f"Board {N}x{N}"]

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quoridor")
    font = pygame.font.SysFont("segoeui", 26, bold=True)
    small = pygame.font.SysFont("segoeui", 18)
    clock = pygame.time.Clock()

    game = Quoridor()
    ai = AI(session, args.sims)
    legal = game.get_legal_actions()

    def reset_to(history):
        nonlocal game, legal
        game = Quoridor()
        for action in history:
            game.do_action(action)
        legal = game.get_legal_actions()
        ai.eval = ai.result = None

    def play(action):
        nonlocal legal
        game.do_action(action)
        legal = game.get_legal_actions() if game.check_win() == -2 else legal[:0]

    while True:
        hover = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and not ai.busy:
                if event.key == pygame.K_n:
                    reset_to([])
                elif event.key == pygame.K_u and game.action_history:
                    history = list(game.action_history)
                    history.pop()
                    while history and len(humans) == 1 and (-1 if len(history) % 2 == 0 else 1) not in humans:
                        history.pop()
                    reset_to(history)
            if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not ai.busy
                    and game.next_player in humans and game.check_win() == -2):
                wall = hovered_wall(event.pos)
                square = hovered_square(event.pos)
                legal_set = {tuple(int(v) for v in a) for a in legal}
                if wall is not None and wall in legal_set:
                    play(np.array(wall, dtype=np.uint8))
                elif wall is None and square is not None and (0, *square) in legal_set:
                    play(np.array((0, *square), dtype=np.uint8))

        if ai.result is not None and not ai.busy:
            action, ai.result = ai.result, None
            play(action)
        if not ai.busy and game.check_win() == -2 and game.next_player not in humans:
            ai.start(game)

        hover = hovered_wall(pygame.mouse.get_pos())
        draw(screen, font, small, game, humans, legal, hover, ai, info)
        clock.tick(30)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
