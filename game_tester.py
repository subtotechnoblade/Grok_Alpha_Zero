import numpy as np
from Guide import Gomoku

class Game_Test:
    def __init__(self, game_class):
        self.game_class: Gomoku = game_class

    def check_legal_actions(self):


    def do_action_test(self):
        if self.game_class.do_action()