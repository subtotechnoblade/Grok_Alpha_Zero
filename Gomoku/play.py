import pygame
import numpy as np
from typing import overload
from glob import glob
from tqdm import tqdm

import onnxruntime as rt
from MCTS_Gumbel import MCTS_Gumbel
from MCTS import MCTS

from Gomoku import Gomoku

pygame.init()

# import ctypes

# ctypes.windll.user32.SetProcessDPIAware()

screen = pygame.display.set_mode((800, 800), vsync=True,
                                 flags=pygame.DOUBLEBUF | pygame.HWACCEL | pygame.RESIZABLE)
LIME_GREEN = (120, 190, 33)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

tile_size = 800 / (15 + 1)

background = pygame.image.load('background.jpeg').convert()


class Button:
    def __init__(self, rect, return_value, color:tuple = (255, 255, 255), text: str = "", text_size=20):
        self.start_x, self.start_y, self.width, self.height = rect
        self.return_value = return_value
        self.color = color
        self.rect = pygame.Rect(rect)
        font = pygame.font.Font('freesansbold.ttf', text_size)
        self.text = font.render(str(text), True, BLACK)

    def Get_info(self):
        return self.rect, self.color, self.text

    def Clicked(self, mouse_x, mouse_y):
        if self.start_x <= mouse_x <= self.start_x + self.width and self.start_y <= mouse_y <= self.start_y + self.height:
            return self.return_value
        return False


class Button_Manager:
    def __init__(self, screen):
        self.screen = screen
        self.buttons = set()
        pass

    def Add_button(self, new_button: Button):
        assert isinstance(new_button, Button)
        assert new_button not in self.buttons
        self.buttons.add(new_button)

    def Render_button(self):
        for button in self.buttons:
            pygame.draw.rect(self.screen, button.color, button.rect)

    def Check_clicked(self, mouse_coords):
        mouse_x, mouse_y = mouse_coords
        for button in self.buttons:
            if return_value := button.Clicked(mouse_x, mouse_y):
                return return_value
        return None

    # @overload
    # def Check_clicked(self, mouse_x, mouse_y):
    #     for button in self.buttons:
    #         if return_value := button.Clicked(mouse_x, mouse_y):
    #             return return_value
    #     return None



def draw_background():
    screen.blit(background, (0, 0))


def draw_board():
    for y in range(15 + 1):
        pygame.draw.line(screen, BLACK, (tile_size, y * tile_size - 1), (15 * tile_size, y * tile_size - 1),
                         width=2)
    for x in range(15 + 1):
        pygame.draw.line(screen, BLACK, (x * tile_size - 1, tile_size), (x * tile_size - 1, (15 * tile_size)),
                         width=2)

    # Centre circle
    pygame.draw.circle(screen, BLACK, (400, 400), 10)

    for x in range(2):
        pygame.draw.circle(screen, BLACK, (200 + 400 * x, 200), 7)
    for x in range(2):
        pygame.draw.circle(screen, BLACK, (200 + 400 * x, 600), 7)


def draw_move(moves):
    if not moves:
        return
    x, y = moves[-1]
    pygame.draw.circle(screen, LIME_GREEN, (tile_size * (x + 1), tile_size * (y + 1)), 30)
    font = pygame.font.Font('freesansbold.ttf', 20)

    for move_num, move in enumerate(moves):
        move_num += 1
        x, y = move
        if move_num % 2 == 0:
            pygame.draw.circle(screen, WHITE, (tile_size * (x + 1), tile_size * (y + 1)), 20)
            render_move_num = font.render(str(move_num), True, BLACK)
            screen.blit(render_move_num, (tile_size * (x + 1) - 11, tile_size * (y + 1) - 10))
        else:
            pygame.draw.circle(screen, BLACK, (tile_size * (x + 1), tile_size * (y + 1)), 20)
            render_move_num = font.render(str(move_num), True, WHITE)
            screen.blit(render_move_num, (tile_size * (x + 1) - 11, tile_size * (y + 1) - 10))

    # pygame.display.update()


coord_list = []
for i in range(15):
    temp_list = []
    y = i * tile_size + 25
    for o in range(15):
        x = o * tile_size + 25
        temp_list.append([x, y, x + tile_size, y + tile_size])
    coord_list.append(temp_list)


def human_move(position, input_board):
    board = input_board.copy()
    cursor_x, cursor_y = position
    for y_move, row in enumerate(coord_list):
        for x_move, (start_x, start_y, end_x, end_y) in enumerate(row):
            if start_y <= cursor_y <= end_y and start_x <= cursor_x <= end_x:
                if board[y_move][x_move] == 0:
                    return x_move, y_move
                return None, None
    return None, None


w, h = 15, 15

game = Gomoku()
# game.put((6, 6))
# game.put((7, 7))
# game.put((8, 6))
# game.put((7, 6))
# game.put((7, 5))
# game.put((5, 7))
# game.put((6, 7))
# game.put((6, 5))
# game.put((8, 7))
# game.put((8, 5))
# game.put((8, 9))
# game.put((7, 8))
# game.put((8, 10))
# game.put((8, 8))
# game.put((8, 4))
# game.put((6, 8))
# game.put((9, 3))
# game.put((10, 2))
# game.put((9, 8))
# game.put((7, 10))
# game.put((7, 9))
# game.put((5, 8))
# game.put((4, 8))
# game.put((6, 9))
#
# game.put((7, 7))
# game.put((5, 7))
# game.put((8, 6))
# game.put((6, 8))
# game.put((7,9))
# game.put((6, 6))
# game.put((7, 5))
# game.put((7, 8))
# game.put((9, 7))
# game.put((10, 8))
# game.put((6, 4))
# game.put((5, 3))
# game.put((8, 8))
# game.put((6, 10))
# game.put((8, 7))
# game.put((6, 7))
# game.put((6, 9))
# game.put((5, 6))


providers = [
    ('TensorrtExecutionProvider', {
        # "trt_engine_cache_enable": True,
        # "trt_dump_ep_context_model": True,
        # "trt_builder_optimization_level": 5,
        # "trt_auxiliary_streams": 0,
        # "trt_ep_context_file_path": "Gomoku/Cache/",
        #
        # "trt_profile_min_shapes": f"inputs:1x15x15,input_state:{num_layers}x2x1x{embed_size},input_state_matrix:{num_layers}x1x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_max_shapes": f"inputs:{max_shape}x15x15,input_state:{num_layers}x2x{max_shape}x{embed_size},input_state_matrix:{num_layers}x{max_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
        # "trt_profile_opt_shapes": f"inputs:{opt_shape}x15x15,input_state:{num_layers}x2x{opt_shape}x{embed_size},input_state_matrix:{num_layers}x{opt_shape}x{num_heads}x{embed_size // num_heads}x{embed_size // num_heads}",
    }),
    # 'CUDAExecutionProvider',
    'CPUExecutionProvider'
]
session = rt.InferenceSession("Grok_Zero_Train/6/TRT_cache/model_ctx.onnx", providers=providers)
# session = rt.InferenceSession("Grok_Zero_Train/3/model.onnx")
# mcts = MCTS_Gumbel(game,
#                     # None,
#                     session,
#                     False,
#                     None,
#                     m=64,
#                     c_visit=50.0,
#                     c_scale=1.0,
#                     fast_find_win=False,
#                     activation_fn="softmax")
mcts = MCTS(game,
            # None,
            session,
            None,
            c_puct_init=2.5,
            tau=0.0,
            dirichlet_alpha=0.333,
            use_dirichlet=False,
            fast_find_win=False)

total_iterations = 2000
time_limit = None

move_list = []

# button1 = Button((500, 500, 100, 50), "hi",)
# button_manager = Button_Manager(screen=screen)
# button_manager.Add_button(button1)


mode = 1
# mode 3, human_player_move = 1, for human against human

# mode 1 for the bottom
human_player_move = 1
# 1 for human becoming the first move
# 2 for human to become the second move
# 3 for AI to play against itself

# I no longer know hos this piece of code works, and I probably never will
# When this code only God and I knew how it worked
# Now only God knows
if mode == 1:
    if human_player_move == 1:
        move_player_black = 0
        move_player_white = 1
    elif human_player_move == 2:
        move_player_black = 0
        move_player_white = 0
    elif human_player_move == 3:
        move_player_black = 0
        move_player_white = 0
else:
    if human_player_move == 1:
        move_player_black = 0
        move_player_white = 0

clock = pygame.time.Clock()

fast = True
ponder = False
turn = len(game.action_history)
player_that_won = 0
won_player = -2
enter_pressed = 0
iterations = 0

shown_result = False

# clock = pygame.time.Clock()
running = True
while running:
    clock.tick(70)
    # draw the stuff, background and board
    screen.fill((255, 255, 255))
    draw_background()
    draw_board()

    # button_manager.Render_button()
    # if return_value := button_manager.Check_clicked(pygame.mouse.get_pos()):
    #     print(return_value)

    pygame_event = pygame.event.get()
    if won_player == -2:
        move = (None, None)
        current_player = game.get_next_player()
        if current_player == -1 and won_player == -2:
            if (1 if current_player == -1 else 2) % human_player_move == move_player_black:
                if pygame.mouse.get_pressed()[0]:
                    move = human_move(pygame.mouse.get_pos(), game.board)
                    # move, probs = mcts.run(iteration_limit=225)
                    # print(probs)

            else:
                # while True:
                #     try:
                #         total_iterations = input("Iterations?").replace(" ", "")
                #         if total_iterations == "":
                #             break
                #         total_iterations = int(total_iterations)
                #     except:
                #         print("Try again")
                #         continue
                #
                #     if total_iterations <= 0:
                #         print(f"Iterations can't be less than 0: {total_iterations}")
                #     else:
                #         break
                if fast:
                    move, line = mcts.run(iteration_limit=total_iterations, time_limit=time_limit)
                    print(line)
                else:
                    if iterations == 0:
                        bar = tqdm(total=total_iterations)
                    if iterations <= total_iterations:
                        mcts.iteration_step()
                        iterations += 1
                        bar.update(1)
                    elif iterations >= total_iterations:
                        move, line = mcts.run(iteration_limit=0)
                        iterations = 0
                        bar.close()

        elif current_player == 1 and won_player == -2:
            if (1 if current_player == -1 else 2) % human_player_move == move_player_white:
                if pygame.mouse.get_pressed()[0]:
                    # mcts.run(iteration_limit=100, use_bar=False)
                    move = human_move(pygame.mouse.get_pos(), game.board)
            else:
                # while True:
                #     try:
                #         total_iterations = input("Iterations?").replace(" ", "")
                #         if total_iterations == "":
                #             break
                #         total_iterations = int(total_iterations)
                #     except:
                #         print("Try again")
                #         continue
                #
                #     if total_iterations <= 0:
                #         print(f"Iterations can't be less than 0: {total_iterations}")
                #     else:
                #         break
                if fast:
                    move, line = mcts.run(iteration_limit=total_iterations, time_limit=time_limit)
                    print(line)
                else:
                    if iterations == 0:
                        bar = tqdm(total=total_iterations)
                    if iterations <= total_iterations:
                        mcts.iteration_step()
                        iterations += 1
                        bar.update(1)
                    elif iterations >= total_iterations:
                        move, line = mcts.run(iteration_limit=0)
                        iterations = 0
                        bar.close()

    if (*move,) != (None, None) and won_player == -2:
        print(move)
        game.do_action(move)
        if mode != 3:
            print("Pruned")
            mcts.prune_tree(move)
        won_player = game.check_win()
        turn += 1

    # if fast and ponder and len(game.moves) != 0:
    #     if human_player_move == 1 and gf.get_next_player(np.array(game.board)) == -1:  # if the human player is 1
    #         mcts.iteration_step()
    #     elif human_player_move == 1 and gf.get_next_player(np.array(game.board)) == 1:
    #         mcts.iteration_step()

    if won_player != -2 and not shown_result:
        if won_player == -1:
            print("BLACK has won")
        elif won_player == 1:
            print("WHITE has won")
        elif won_player == 0:
            print("GAME was a draw")
        shown_result = True

    # if len(game.moves) == 1:
    #     draw_move(game.moves)
    # # if len(game.moves) != 0 and len(game.moves) == 1:
    # #     draw_move(game.moves)
    # elif len(game.moves) > 1:
    draw_move(game.action_history[:turn])

    keys = pygame.key.get_pressed()
    for event in pygame_event:
        if event.type == pygame.KEYDOWN:
            if keys[pygame.K_RETURN]:
                enter_pressed += 1
                # if enter_pressed % 1 == 1:
                #     game_number = len(glob("played_games/*.psq"))
                #     with open(f"played_games/{game_number}.psq", 'a') as data:
                #         for x, y in move_list:
                #             data.write(f"\n{str(x)},{str(y)}")
                if enter_pressed % 2 == 1:
                    print(game.action_history)
                    # reset the game
                    player_that_won = 0
                    won_player = -2
                    shown_result = False
                    # reset the board and move list
                    game = Gomoku()
                    mcts = MCTS_Gumbel(game,
                                        # None,
                                        session,
                                        True,
                                        None,
                                        m=16,
                                        c_visit=50.0,
                                        c_scale=0.15,
                                        fast_find_win=False,
                                        activation_fn="softmax")



            elif keys[pygame.K_LEFT]:
                if turn > 0:
                    turn -= 1
            elif keys[pygame.K_RIGHT]:
                if turn < len(game.action_history):
                    turn += 1

    # print(pygame.mouse.get_pos())
    pygame.display.update()

    # pygame.display.flip()
    for event in pygame_event:
        if event.type == pygame.QUIT:
            running = False
    # clock.tick(65)
pygame.quit()
