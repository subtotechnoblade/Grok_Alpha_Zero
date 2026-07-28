"""Pygame front end for the Connect 4 agent.

Run it directly:  python Connect4/play.py     (works from any working directory)

Modes: play as red, play as yellow, watch the net play itself, or two humans.
The right hand panel shows what the search actually thinks: the root value as a
win probability, and the visit distribution vs the raw policy prior per column,
which is the quickest way to see where the network disagrees with the search.
"""

import os
import sys
import types
import math
import glob
import threading


# --------------------------------------------------------------------------- paths
# only the repo root goes on the path. This folder used to go on it too, which let
# game.py be imported as either `Connect4` or `Connect4.game` depending on who
# launched, giving numba two module identities for one file
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------- numba
def _install_numba_stub() -> bool:
    """Return True if the real numba works, otherwise stub it out.

    Connect4.py and MCTS.py both `from numba import njit` at module scope, and the
    Anaconda install on Windows ships a numpy that its numba refuses (it wants
    <= 1.21), so the import dies before the window ever opens. MCTS already has a
    pure python path -- it checks is_jitted() on the game methods and falls back to
    the plain get_terminal_actions_fn -- so a no-op decorator is enough to run.
    Inference dominates the per-move cost anyway; this is not the bottleneck.
    """
    try:
        import numba  # noqa: F401
        from numba.extending import is_jitted  # noqa: F401
        return True
    except Exception:
        pass

    def njit(*args, **kwargs):
        # supports @njit, @njit("sig", cache=True) and njit(fn, cache=True)
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    numba = types.ModuleType("numba")
    numba.njit = njit
    numba.jit = njit
    extending = types.ModuleType("numba.extending")
    extending.is_jitted = lambda fn: False
    numba.extending = extending
    sys.modules["numba"] = numba
    sys.modules["numba.extending"] = extending
    return False


NUMBA_OK = _install_numba_stub()

import numpy as np
import pygame

from Connect4.game import Connect4
from MCTS import MCTS


# --------------------------------------------------------------------------- layout
COLS, ROWS = 7, 6
CELL = 90
DISC = 72
PAD = 14

BOARD_W = COLS * CELL + 2 * PAD          # 658
BOARD_H = ROWS * CELL + 2 * PAD          # 568
BOARD_X = 36
BOARD_Y = 140

WINDOW_W, WINDOW_H = 1120, 780
PANEL_X = BOARD_X + BOARD_W + 30         # 724
PANEL_W = WINDOW_W - PANEL_X - 32        # 364

HOVER_CY = BOARD_Y - 44                  # centre of the floating "about to drop" disc

RED, YELLOW = -1, 1                      # Connect4 uses -1 as the first player


# --------------------------------------------------------------------------- palette
BG_TOP      = (12, 16, 32)
BG_BOT      = (24, 30, 56)
PLATE_TOP   = (48, 96, 236)
PLATE_BOT   = (22, 50, 156)
HOLE_RING   = (10, 20, 62, 255)
GLOW        = (60, 120, 255)

DISC_COLS = {
    RED:    ((255, 150, 152), (233, 60, 72), (140, 16, 30)),
    YELLOW: ((255, 235, 158), (247, 193, 40), (158, 104, 6)),
}

TEXT      = (233, 238, 252)
MUTED     = (136, 149, 186)
DIM       = (86, 98, 134)
CARD      = (21, 28, 51)
CARD_EDGE = (43, 55, 92)
ACCENT    = (96, 182, 255)
GOOD      = (86, 214, 148)


# --------------------------------------------------------------------------- helpers
def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    if len(a) == 4 or len(b) == 4:
        a = tuple(a) + (255,) * (4 - len(a))
        b = tuple(b) + (255,) * (4 - len(b))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def ease_out(t):
    return 1.0 - (1.0 - t) ** 3


def vgradient(size, top, bottom):
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        pygame.draw.line(surf, mix(top, bottom, y / max(1, h - 1)), (0, y), (w, y))
    return surf


def cell_center(row, col):
    return (BOARD_X + PAD + col * CELL + CELL // 2,
            BOARD_Y + PAD + row * CELL + CELL // 2)


def find_win_line(board, player):
    """The four cells that won, for the highlight. check_win only returns who."""
    for y in range(ROWS):
        for x in range(COLS):
            for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                cells = []
                for k in range(4):
                    ny, nx = y + dy * k, x + dx * k
                    if not (0 <= ny < ROWS and 0 <= nx < COLS):
                        break
                    if board[ny][nx] != player:
                        break
                    cells.append((ny, nx))
                if len(cells) == 4:
                    return cells
    return None


# --------------------------------------------------------------------------- sprites
def make_disc(diameter, light, mid, dark):
    """A shaded disc. Rendered 4x and downscaled, which is how the edges get smooth."""
    ss = 4
    n = diameter * ss
    r = n // 2
    surf = pygame.Surface((n, n), pygame.SRCALPHA)

    for i in range(r, 0, -1):
        t = i / r                                    # 1 at the rim, 0 at the centre
        col = mix(light, mid, t ** 0.9) if t < 0.72 else mix(mid, dark, (t - 0.72) / 0.28)
        # drift the centre up-left as the radius shrinks -> off centre highlight
        cx = r - int(0.20 * (r - i))
        cy = r - int(0.24 * (r - i))
        pygame.draw.circle(surf, col, (cx, cy), i)

    pygame.draw.circle(surf, dark, (r, r), r, width=max(ss, r // 13))

    # soft specular, added in RGB only so it cannot bleed outside the disc
    spec = pygame.Surface((n, n), pygame.SRCALPHA)
    steps = 9
    for k in range(steps, 0, -1):
        f = k / steps
        ew, eh = r * 0.86 * f, r * 0.56 * f
        ecx, ecy = r * 0.70, r * 0.58
        v = int(10 + 30 * (1 - f))
        pygame.draw.ellipse(spec, (v, v, v), (ecx - ew / 2, ecy - eh / 2, ew, eh))
    surf.blit(spec, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    return pygame.transform.smoothscale(surf, (diameter, diameter))


def make_plate():
    """The blue front panel, with the holes punched straight out of it.

    pygame's draw functions write pixels rather than blending, so drawing a
    fully transparent circle really does cut a hole in an SRCALPHA surface.
    """
    ss = 3
    w, h = BOARD_W * ss, BOARD_H * ss
    plate = vgradient((w, h), PLATE_TOP, PLATE_BOT)

    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=26 * ss)
    plate.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    # sheen fading down from the top edge; RGB only, so the rounded corners stay clipped
    gloss = pygame.Surface((w, h), pygame.SRCALPHA)
    band = 44 * ss
    for y in range(band):
        v = int(24 * (1 - y / band) ** 1.7)
        if v:
            pygame.draw.line(gloss, (v, v, v), (0, y), (w, y))
    plate.blit(gloss, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    hole_r = (DISC // 2 + 2) * ss
    for row in range(ROWS):
        for col in range(COLS):
            cx = (PAD + col * CELL + CELL // 2) * ss
            cy = (PAD + row * CELL + CELL // 2) * ss
            pygame.draw.circle(plate, HOLE_RING, (cx, cy), hole_r + 3 * ss)
            pygame.draw.circle(plate, (0, 0, 0, 0), (cx, cy), hole_r)

    return pygame.transform.smoothscale(plate, (BOARD_W, BOARD_H))


def make_shadow(diameter, spread=10):
    n = diameter + spread * 2
    surf = pygame.Surface((n, n), pygame.SRCALPHA)
    for i in range(spread, 0, -1):
        a = int(70 * (1 - i / spread) ** 2)
        pygame.draw.circle(surf, (0, 0, 0, a), (n // 2, n // 2), diameter // 2 + i)
    return surf


# --------------------------------------------------------------------------- engine
def available_generations():
    found = []
    for path in glob.glob(os.path.join(_HERE, "Grok_Zero_Train", "*", "model.onnx")):
        try:
            found.append(int(os.path.basename(os.path.dirname(path))))
        except ValueError:
            pass
    return sorted(found)


class Brain:
    """Owns the onnx session and the search tree, and runs the search off-thread."""

    CHUNK = 48   # sims per run() call, so the progress bar actually moves

    def __init__(self, game, generation, sims):
        self.generation = generation
        self.sims = sims
        self.session = self._load(generation)

        self.mcts = MCTS(game,
                         self.session,
                         use_njit=NUMBA_OK,
                         c_puct_init=1.5,
                         use_dirichlet=False,   # no exploration noise when playing for real
                         tau=0.0,               # always take the most visited move
                         fast_find_win=True)

        self._thread = None
        self.result = None
        self.progress = 0.0

        self._warm = threading.Thread(target=self._warmup, daemon=True)
        self._warm.start()

    def _warmup(self):
        """Touch every numba specialisation the search will need, on a throwaway tree.

        The first call into each jitted signature loads a cached compile off disk, and
        under WSL the repo sits on /mnt/c (a 9P mount) where that takes seconds. _expand
        hits signatures create_expand_root never does, so without this the very first
        engine move is the one that pays. Uses its own game and MCTS so the real tree
        does not pick up phantom visits.
        """
        try:
            scratch = Connect4()
            tree = MCTS(scratch, self.session, use_njit=NUMBA_OK,
                        use_dirichlet=False, tau=0.0, fast_find_win=True)
            tree.run(16, use_bar=False)
            scratch.do_action(3)
            tree.prune_tree(3)
            tree.run(16, use_bar=False)
        except Exception as exc:
            print(f"[warmup] {type(exc).__name__}: {exc}")

    @property
    def warming(self):
        return self._warm is not None and self._warm.is_alive()

    @staticmethod
    def _load(generation):
        if generation is None:
            return None
        import onnxruntime as rt
        path = os.path.join(_HERE, "Grok_Zero_Train", str(generation), "model.onnx")
        if not os.path.exists(path):
            return None
        avail = rt.get_available_providers()
        # skip TensorRT on purpose: building an engine takes minutes and this is a GUI
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        opts = rt.SessionOptions()
        opts.log_severity_level = 3
        return rt.InferenceSession(path, sess_options=opts, providers=providers or None)

    @property
    def thinking(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        self.result = None
        self.progress = 0.0
        self._thread = threading.Thread(target=self._search, daemon=True)
        self._thread.start()

    def _search(self):
        done = 0
        move, stats = None, None
        try:
            while done < self.sims:
                chunk = min(self.CHUNK, self.sims - done)
                move, stats = self.mcts.run(chunk, use_bar=False)
                done += chunk
                self.progress = done / self.sims
        except Exception as exc:          # never let a search bug kill the window
            print(f"[search error] {type(exc).__name__}: {exc}")
            legal = self.mcts.game.get_legal_actions()
            move, stats = int(legal[0]), []
        self.result = (int(move), stats)

    def resync(self, action=None):
        """Keep the tree pointed at the real position after any move or undo."""
        if action is None:
            self.mcts.prune_tree(None, create_new_root=True)
        else:
            self.mcts.prune_tree(action)


# --------------------------------------------------------------------------- widgets
class Button:
    def __init__(self, rect, label, hint=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.hint = hint
        self.enabled = True

    def draw(self, surf, fonts, mouse):
        hot = self.enabled and self.rect.collidepoint(mouse)
        base = mix(CARD, ACCENT, 0.22) if hot else CARD
        if not self.enabled:
            base = mix(CARD, BG_TOP, 0.5)
        pygame.draw.rect(surf, base, self.rect, border_radius=11)
        pygame.draw.rect(surf, mix(CARD_EDGE, ACCENT, 0.5 if hot else 0.0),
                         self.rect, width=1, border_radius=11)

        colour = TEXT if self.enabled else DIM
        label = fonts["ui"].render(self.label, True, colour)
        if self.hint:
            surf.blit(label, label.get_rect(midleft=(self.rect.x + 14, self.rect.centery)))
            hint = fonts["small"].render(self.hint, True, DIM if self.enabled else DIM)
            surf.blit(hint, hint.get_rect(midright=(self.rect.right - 14, self.rect.centery)))
        else:
            surf.blit(label, label.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


class Drop:
    """Gravity plus a couple of damped bounces."""
    GRAVITY = 3400.0

    def __init__(self, col, row, player, from_y):
        self.col = col
        self.row = row
        self.player = player
        self.y = float(from_y)
        self.vy = 0.0
        self.target = float(cell_center(row, col)[1])
        self.bounces = 0
        self.done = False

    def step(self, dt):
        self.vy += self.GRAVITY * dt
        self.y += self.vy * dt
        if self.y >= self.target:
            self.y = self.target
            if self.bounces >= 2 or self.vy < 220:
                self.done = True
            else:
                self.vy = -self.vy * 0.28
                self.bounces += 1


# --------------------------------------------------------------------------- the app
MENU, PLAY, OVER = "menu", "play", "over"


class Connect4GUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Connect 4  --  AlphaZero")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.fonts = {
            "title": pygame.font.SysFont("segoeui,dejavusans,arial", 34, bold=True),
            "h1":    pygame.font.SysFont("segoeui,dejavusans,arial", 23, bold=True),
            "ui":    pygame.font.SysFont("segoeui,dejavusans,arial", 17),
            "small": pygame.font.SysFont("segoeui,dejavusans,arial", 14),
            "mono":  pygame.font.SysFont("consolas,dejavusansmono,couriernew", 14),
            "huge":  pygame.font.SysFont("segoeui,dejavusans,arial", 44, bold=True),
        }

        self.bg = self._make_background()
        self.glow = self._make_glow()
        self.plate = make_plate()
        self.discs = {p: make_disc(DISC, *DISC_COLS[p]) for p in (RED, YELLOW)}
        self.ghosts = {}
        for p in (RED, YELLOW):
            g = self.discs[p].copy()
            g.set_alpha(90)
            self.ghosts[p] = g
        self.shadow = make_shadow(DISC)

        self.generations = available_generations()
        self.gen_index = len(self.generations) - 1
        self.sims = 400          # ~1.4s per 100 sims on CPU without numba; tune in the menu
        self.human_side = RED
        self.two_players = False
        self.ai_vs_ai = False

        self.game = Connect4()
        self.brain = None
        self.state = MENU

        self.anim = None
        self.winner = -2
        self.win_line = None
        self.last_move = None
        self.stats = []
        self.stats_mover = None      # which side the displayed search was for
        self.has_eval = False
        self.eval_red = 0.5          # displayed, eased toward eval_target
        self.eval_target = 0.5
        self.time_ms = 0
        self.think_started = 0

        self.buttons = {
            "undo":    Button((PANEL_X, 650, PANEL_W // 2 - 6, 44), "Undo", "U"),
            "restart": Button((PANEL_X + PANEL_W // 2 + 6, 650, PANEL_W // 2 - 6, 44), "New", "R"),
            "menu":    Button((PANEL_X, 704, PANEL_W, 44), "Main menu", "Esc"),
        }
        self.menu_buttons = {}

    # ------------------------------------------------------------------ assets
    def _make_background(self):
        return vgradient((WINDOW_W, WINDOW_H), BG_TOP, BG_BOT)

    def _make_glow(self):
        """Halo behind the board. Kept off the background so the menu stays clean."""
        glow = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        cx, cy = BOARD_X + BOARD_W // 2, BOARD_Y + BOARD_H // 2
        for i in range(26, 0, -1):
            a = int(3.4 * (1 - i / 26) ** 1.6 * 26)
            pygame.draw.rect(glow, (*GLOW, a),
                             (cx - BOARD_W // 2 - i * 5, cy - BOARD_H // 2 - i * 5,
                              BOARD_W + i * 10, BOARD_H + i * 10),
                             border_radius=26 + i * 4)
        return glow

    # ------------------------------------------------------------------ game flow
    def new_game(self):
        self.game.board[:] = 0
        self.game.next_player = RED
        self.game.action_history.clear()
        self.anim = None
        self.winner = -2
        self.win_line = None
        self.last_move = None
        self.stats = []
        self.stats_mover = None
        self.has_eval = False
        self.eval_red = self.eval_target = 0.5
        self.time_ms = 0
        self.state = PLAY

        gen = self.generations[self.gen_index] if self.generations else None
        if self.brain is None or self.brain.generation != gen or self.brain.sims != self.sims:
            self.brain = Brain(self.game, gen, self.sims)
        else:
            self.brain.resync()

    def ai_to_move(self):
        if self.two_players:
            return False
        if self.ai_vs_ai:
            return True
        return self.game.get_next_player() != self.human_side

    def play_action(self, action):
        action = int(action)
        col = self.game.board[:, action]
        row = 5 - int(np.sum(np.abs(col)))
        if row < 0:
            return
        player = self.game.get_next_player()
        self.game.do_action(action)
        self.last_move = (row, action)
        self.anim = Drop(action, row, player, from_y=HOVER_CY)

    def _on_landed(self):
        action = self.game.action_history[-1]
        self.anim = None
        self.winner = int(self.game.check_win())
        if self.winner != -2:
            self.state = OVER
            if self.winner != 0:
                self.win_line = find_win_line(self.game.board, self.winner)
            return
        self.brain.resync(action)

    def undo(self):
        if self.brain.thinking or self.anim or not self.game.action_history:
            return
        history = list(self.game.action_history[:-1])
        if not self.two_players and not self.ai_vs_ai and history:
            # step back past the engine's reply so it is the human's turn again
            if (len(history) % 2 == 0) != (self.human_side == RED):
                history = history[:-1]

        self.game.board[:] = 0
        self.game.next_player = RED
        self.game.action_history.clear()
        for a in history:
            self.game.do_action(a)

        self.winner = -2
        self.win_line = None
        if history:
            col = history[-1]
            # topmost disc in that column sits at row 6 - height
            self.last_move = (ROWS - int(np.sum(np.abs(self.game.board[:, col]))), col)
        else:
            self.last_move = None
        self.stats = []
        self.stats_mover = None
        self.has_eval = False
        self.eval_target = 0.5
        self.state = PLAY
        self.brain.resync()

    def hovered_column(self, mouse):
        mx, my = mouse
        if not (BOARD_X <= mx <= BOARD_X + BOARD_W and HOVER_CY - 40 <= my <= BOARD_Y + BOARD_H):
            return None
        col = (mx - BOARD_X - PAD) // CELL
        if 0 <= col < COLS and int(np.sum(np.abs(self.game.board[:, col]))) < ROWS:
            return int(col)
        return None

    # ------------------------------------------------------------------ update
    def update(self, dt, mouse):
        if self.state == MENU:
            return

        if self.anim is not None:
            self.anim.step(dt)
            if self.anim.done:
                self._on_landed()
            return

        if self.state == OVER:
            return

        if self.ai_to_move():
            if not self.brain.thinking and self.brain.result is None:
                self.think_started = pygame.time.get_ticks()
                self.brain.start()
            elif self.brain.result is not None:
                action, stats = self.brain.result
                self.brain.result = None
                self.time_ms = pygame.time.get_ticks() - self.think_started
                self.stats = stats
                self._read_eval(stats)
                self.play_action(action)

        self.eval_red += (self.eval_target - self.eval_red) * min(1.0, dt * 6.0)

    def _read_eval(self, stats):
        """stats rows are [action, visit%, winrate, value, visits, prior, root_visits, terminal].

        winrate is from the point of view of the player to move at the root -- verified
        against a forced win, which comes back as exactly 1.0. Terminal wins back up
        1.0 and draws 0.0, so the usable range is [-1, 1]; map it onto a red-side bar.
        """
        if not stats:
            return
        winrate = float(stats[0][2])
        p_win = max(0.0, min(1.0, (winrate + 1.0) / 2.0))
        mover = self.game.get_next_player()
        self.eval_target = p_win if mover == RED else 1.0 - p_win
        self.stats_mover = mover
        self.has_eval = True

    # ------------------------------------------------------------------ drawing
    def draw(self, mouse):
        self.screen.blit(self.bg, (0, 0))
        if self.state == MENU:
            self.draw_menu(mouse)
        else:
            self.screen.blit(self.glow, (0, 0))
            self.draw_header()
            self.draw_board(mouse)
            self.draw_panel(mouse)
        pygame.display.flip()

    def _card(self, rect, label=None):
        pygame.draw.rect(self.screen, CARD, rect, border_radius=13)
        pygame.draw.rect(self.screen, CARD_EDGE, rect, width=1, border_radius=13)
        if label:
            txt = self.fonts["small"].render(label.upper(), True, DIM)
            self.screen.blit(txt, (rect[0] + 15, rect[1] + 12))

    def draw_header(self):
        self.screen.blit(self.fonts["title"].render("Connect 4", True, TEXT), (BOARD_X, 26))
        if self.generations:
            gen = self.generations[self.gen_index]
            sub = f"generation {gen}   ·   {self.sims} simulations"
        else:
            sub = "no trained model found  ·  search only"
        if not NUMBA_OK:
            sub += "   ·   numba off"
        self.screen.blit(self.fonts["small"].render(sub, True, MUTED), (BOARD_X + 2, 70))

    def draw_board(self, mouse):
        hover = None
        if self.state == PLAY and self.anim is None and not self.ai_to_move():
            hover = self.hovered_column(mouse)

        # column beam, kept inside the board so it does not box in the floating ghost
        if hover is not None:
            h = BOARD_H - 2 * PAD
            beam = pygame.Surface((CELL, h), pygame.SRCALPHA)
            for y in range(h):
                pygame.draw.line(beam, (255, 255, 255, int(30 * (1 - y / h)) + 5),
                                 (0, y), (CELL, y))
            self.screen.blit(beam, (BOARD_X + PAD + hover * CELL, BOARD_Y + PAD))

        # settled discs, behind the plate so the holes frame them
        for row in range(ROWS):
            for col in range(COLS):
                p = int(self.game.board[row][col])
                if p == 0:
                    continue
                if self.anim is not None and (row, col) == (self.anim.row, self.anim.col):
                    continue
                cx, cy = cell_center(row, col)
                self.screen.blit(self.shadow, self.shadow.get_rect(center=(cx, cy + 3)))
                self.screen.blit(self.discs[p], self.discs[p].get_rect(center=(cx, cy)))

        # falling disc
        if self.anim is not None:
            cx = cell_center(self.anim.row, self.anim.col)[0]
            sprite = self.discs[self.anim.player]
            self.screen.blit(self.shadow, self.shadow.get_rect(center=(cx, int(self.anim.y) + 3)))
            self.screen.blit(sprite, sprite.get_rect(center=(cx, int(self.anim.y))))

        # last move marker: a pip in the middle of the disc, so it shows through the hole
        if self.last_move and self.anim is None and not self.win_line:
            row, col = self.last_move
            if 0 <= row < ROWS and int(self.game.board[row][col]) != 0:
                cx, cy = cell_center(row, col)
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 5)
                pygame.draw.circle(self.screen, (40, 20, 20), (cx, cy), 5, width=1)

        # hovering ghost
        if hover is not None:
            g = self.ghosts[self.game.get_next_player()]
            self.screen.blit(g, g.get_rect(center=(BOARD_X + PAD + hover * CELL + CELL // 2, HOVER_CY)))

        self.screen.blit(self.plate, (BOARD_X, BOARD_Y))

        # the winning four, ringed ON TOP of the plate or the hole edge swallows it
        if self.win_line:
            t = (math.sin(pygame.time.get_ticks() / 220.0) + 1) / 2
            r = DISC // 2 + 4
            for (row, col) in self.win_line:
                cx, cy = cell_center(row, col)
                ring = pygame.Surface((r * 2 + 24, r * 2 + 24), pygame.SRCALPHA)
                c = ring.get_width() // 2
                for i in range(7, 0, -1):
                    a = int((40 + 80 * t) * (1 - i / 7))
                    pygame.draw.circle(ring, (255, 255, 255, a), (c, c), r + i, width=3)
                pygame.draw.circle(ring, (255, 255, 255, int(140 + 100 * t)), (c, c), r, width=3)
                self.screen.blit(ring, ring.get_rect(center=(cx, cy)))

        if self.state == OVER:
            self.draw_banner()

    def draw_banner(self):
        if self.winner == 0:
            text, colour = "Draw", MUTED
        elif self.two_players or self.ai_vs_ai:
            text = ("Red" if self.winner == RED else "Yellow") + " wins"
            colour = DISC_COLS[self.winner][1]
        else:
            won = self.winner == self.human_side
            text = "You win" if won else "Engine wins"
            colour = GOOD if won else DISC_COLS[self.winner][1]

        surf = self.fonts["huge"].render(text, True, colour)
        box = surf.get_rect(center=(BOARD_X + BOARD_W // 2, BOARD_Y + BOARD_H // 2))
        pad = pygame.Rect(box.x - 40, box.y - 22, box.w + 80, box.h + 44)
        shade = pygame.Surface(pad.size, pygame.SRCALPHA)
        shade.fill((8, 12, 26, 226))
        self.screen.blit(shade, pad.topleft)
        pygame.draw.rect(self.screen, colour, pad, width=2, border_radius=16)
        self.screen.blit(surf, box)

    def draw_panel(self, mouse):
        # ---- status
        self._card((PANEL_X, 26, PANEL_W, 104), "status")
        if self.state == OVER:
            line = "Game over"
        elif self.brain.thinking:
            dots = "." * (1 + int(pygame.time.get_ticks() / 320) % 3)
            line = f"Thinking{dots}"
        elif self.brain.warming:
            line = "Warming up..."
        elif self.ai_to_move():
            line = "Engine to move"
        else:
            line = "Red to move" if self.game.get_next_player() == RED else "Yellow to move"
        self.screen.blit(self.fonts["h1"].render(line, True, TEXT), (PANEL_X + 15, 54))

        if self.brain.thinking:
            bar = pygame.Rect(PANEL_X + 15, 100, PANEL_W - 30, 6)
            pygame.draw.rect(self.screen, mix(CARD, BG_TOP, 0.6), bar, border_radius=3)
            fill = bar.copy()
            fill.w = max(3, int(bar.w * self.brain.progress))
            pygame.draw.rect(self.screen, ACCENT, fill, border_radius=3)
        else:
            info = f"ply {len(self.game.action_history)}"
            if self.time_ms:
                info += f"   ·   last search {self.time_ms / 1000:.1f}s"
            self.screen.blit(self.fonts["small"].render(info, True, MUTED), (PANEL_X + 16, 96))

        # ---- evaluation
        self._card((PANEL_X, 142, PANEL_W, 104), "evaluation")
        bar = pygame.Rect(PANEL_X + 15, 182, PANEL_W - 30, 26)
        if not self.has_eval:
            pygame.draw.rect(self.screen, mix(CARD, BG_TOP, 0.55), bar, border_radius=13)
        else:
            pygame.draw.rect(self.screen, DISC_COLS[YELLOW][1], bar, border_radius=13)
            red_w = int(bar.w * self.eval_red)
            if red_w > 0:
                # clip so the red side keeps the rounded end of the same pill
                self.screen.set_clip(pygame.Rect(bar.x, bar.y, red_w, bar.h))
                pygame.draw.rect(self.screen, DISC_COLS[RED][1], bar, border_radius=13)
                self.screen.set_clip(None)
            pygame.draw.line(self.screen, (250, 250, 250),
                             (bar.centerx, bar.y + 1), (bar.centerx, bar.bottom - 1), 1)
        pygame.draw.rect(self.screen, CARD_EDGE, bar, width=1, border_radius=13)

        if self.has_eval:
            left = self.fonts["small"].render(f"red {self.eval_red * 100:.0f}%", True, TEXT)
            right = self.fonts["small"].render(f"{(1 - self.eval_red) * 100:.0f}% yellow", True, TEXT)
            self.screen.blit(left, (bar.x, bar.bottom + 8))
            self.screen.blit(right, right.get_rect(topright=(bar.right, bar.bottom + 8)))
        else:
            note = self.fonts["small"].render("no engine evaluation yet", True, DIM)
            self.screen.blit(note, (bar.x, bar.bottom + 8))

        # ---- search detail
        rect = (PANEL_X, 258, PANEL_W, 380)
        side = ""
        if self.stats_mover is not None:
            side = "  ·  " + ("red" if self.stats_mover == RED else "yellow")
        self._card(rect, "search  ·  visits vs prior" + side)
        if self.stats:
            by_col = {int(r[0]): r for r in self.stats}
            top = max((int(r[4]) for r in self.stats), default=1)
            y0 = 298
            row_h = 45
            for col in range(COLS):
                y = y0 + col * row_h
                label = self.fonts["mono"].render(str(col + 1), True, MUTED)
                self.screen.blit(label, (PANEL_X + 16, y + 8))

                track = pygame.Rect(PANEL_X + 38, y + 6, PANEL_W - 104, 13)
                pygame.draw.rect(self.screen, mix(CARD, BG_TOP, 0.55), track, border_radius=6)

                row = by_col.get(col)
                if row is None:
                    continue
                _, share, winrate, _value, visits, prior, _rv, terminal = row
                w = max(2, int(track.w * int(visits) / max(1, top)))
                colour = GOOD if terminal not in (None, 0) else ACCENT
                pygame.draw.rect(self.screen, colour,
                                 (track.x, track.y, w, track.h), border_radius=6)

                # the raw network prior, as a tick under the visit bar
                px = track.x + int(track.w * float(prior))
                pygame.draw.line(self.screen, (255, 255, 255, 150),
                                 (px, track.bottom + 3), (px, track.bottom + 9), 2)

                pct = self.fonts["small"].render(f"{float(share) * 100:4.0f}%", True, TEXT)
                self.screen.blit(pct, pct.get_rect(topright=(rect[0] + rect[2] - 14, y + 4)))
                sub = self.fonts["small"].render(f"q {float(winrate):+.2f}", True, DIM)
                self.screen.blit(sub, (track.x, track.bottom + 10))
            legend = self.fonts["small"].render("bar = visits    tick = policy prior", True, DIM)
            self.screen.blit(legend, (PANEL_X + 16, rect[1] + rect[3] - 24))
        else:
            msg = "the engine has not searched this position"
            note = self.fonts["small"].render(msg, True, DIM)
            self.screen.blit(note, note.get_rect(center=(PANEL_X + PANEL_W // 2,
                                                        rect[1] + rect[3] // 2)))

        # ---- buttons
        self.buttons["undo"].enabled = bool(self.game.action_history) and not self.brain.thinking
        self.buttons["restart"].enabled = True
        for b in self.buttons.values():
            b.draw(self.screen, self.fonts, mouse)

    # ------------------------------------------------------------------ menu
    def draw_menu(self, mouse):
        cx = WINDOW_W // 2
        title = self.fonts["huge"].render("Connect 4", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(cx, 118)))
        sub = self.fonts["ui"].render("play against the network, or watch it play itself",
                                      True, MUTED)
        self.screen.blit(sub, sub.get_rect(center=(cx, 162)))

        # a decorative row of discs
        for i, p in enumerate((RED, YELLOW, RED, YELLOW)):
            d = self.discs[p]
            self.screen.blit(d, d.get_rect(center=(cx - 168 + i * 112, 246)))

        w, h, gap = 420, 54, 14
        x = cx - w // 2
        y = 330
        self.menu_buttons = {}
        entries = [
            ("red",   "Play as Red", "moves first"),
            ("yellow", "Play as Yellow", "moves second"),
            ("watch", "Watch engine self-play", None),
            ("two",   "Two players", "no engine"),
        ]
        for key, label, hint in entries:
            b = Button((x, y, w, h), label, hint)
            b.draw(self.screen, self.fonts, mouse)
            self.menu_buttons[key] = b
            y += h + gap

        # settings row
        y += 12
        gen_txt = f"generation {self.generations[self.gen_index]}" if self.generations else "no model"
        self.menu_buttons["gen_down"] = Button((x, y, 46, 46), "<")
        self.menu_buttons["gen_up"] = Button((x + w - 46, y, 46, 46), ">")
        self.menu_buttons["gen_down"].enabled = bool(self.generations) and self.gen_index > 0
        self.menu_buttons["gen_up"].enabled = bool(self.generations) and self.gen_index < len(self.generations) - 1
        self.menu_buttons["gen_down"].draw(self.screen, self.fonts, mouse)
        self.menu_buttons["gen_up"].draw(self.screen, self.fonts, mouse)
        t = self.fonts["ui"].render(gen_txt, True, TEXT)
        self.screen.blit(t, t.get_rect(center=(cx, y + 23)))

        y += 58
        self.menu_buttons["sim_down"] = Button((x, y, 46, 46), "<")
        self.menu_buttons["sim_up"] = Button((x + w - 46, y, 46, 46), ">")
        self.menu_buttons["sim_down"].enabled = self.sims > 100
        self.menu_buttons["sim_up"].enabled = self.sims < 5000
        self.menu_buttons["sim_down"].draw(self.screen, self.fonts, mouse)
        self.menu_buttons["sim_up"].draw(self.screen, self.fonts, mouse)
        t = self.fonts["ui"].render(f"{self.sims} simulations / move", True, TEXT)
        self.screen.blit(t, t.get_rect(center=(cx, y + 23)))

        foot = "click a column or press 1-7 to drop    ·    U undo    ·    R new game    ·    Esc menu"
        f = self.fonts["small"].render(foot, True, DIM)
        self.screen.blit(f, f.get_rect(center=(cx, WINDOW_H - 40)))

    def menu_click(self, pos):
        hit = [k for k, b in self.menu_buttons.items() if b.hit(pos)]
        if not hit:
            return
        key = hit[0]
        if key == "gen_down":
            self.gen_index = max(0, self.gen_index - 1)
        elif key == "gen_up":
            self.gen_index = min(len(self.generations) - 1, self.gen_index + 1)
        elif key == "sim_down":
            self.sims = max(100, self.sims - 100)
        elif key == "sim_up":
            self.sims = min(5000, self.sims + 100)
        else:
            self.two_players = key == "two"
            self.ai_vs_ai = key == "watch"
            if key in ("red", "yellow"):
                self.human_side = RED if key == "red" else YELLOW
            self.brain = None          # settings may have changed, rebuild
            self.new_game()

    # ------------------------------------------------------------------ loop
    def run(self):
        running = True
        while running:
            dt = min(0.05, self.clock.tick(60) / 1000.0)
            mouse = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = MENU
                    elif self.state != MENU:
                        if event.key == pygame.K_r:
                            self.new_game()
                        elif event.key == pygame.K_u:
                            self.undo()
                        elif pygame.K_1 <= event.key <= pygame.K_7:
                            col = event.key - pygame.K_1
                            if (self.state == PLAY and self.anim is None
                                    and not self.ai_to_move()
                                    and int(np.sum(np.abs(self.game.board[:, col]))) < ROWS):
                                self.play_action(col)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.state == MENU:
                        self.menu_click(event.pos)
                    else:
                        if self.buttons["menu"].hit(event.pos):
                            self.state = MENU
                        elif self.buttons["restart"].hit(event.pos):
                            self.new_game()
                        elif self.buttons["undo"].hit(event.pos):
                            self.undo()
                        elif self.state == PLAY and self.anim is None and not self.ai_to_move():
                            col = self.hovered_column(event.pos)
                            if col is not None:
                                self.play_action(col)

            self.update(dt, mouse)
            self.draw(mouse)

        pygame.quit()


def main():
    Connect4GUI().run()


if __name__ == "__main__":
    main()
