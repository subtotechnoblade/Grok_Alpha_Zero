import numpy as np
from Guide import Gomoku

class Game_Test:
    def __init__(self, game_class):
        self.game_class: Gomoku = game_class

    def current_player_test(self):
        if self.game_class.get_current_player() !=