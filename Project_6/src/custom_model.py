from copy import copy, deepcopy
import numpy as np
import pygame
from piece import BODIES, Piece
from board import Board
from random import randint

"""
Add your code here 
"""

class CUSTOM_AI_MODEL:
    def __init__(self):
        pass

    def get_best_move(self, board, piece, depth=1):
        i = 1
        best_x = 0
        best_piece = piece

        # random rotation
        for x in range(randint(0, 3)):
            best_piece = piece.get_next_rotation()

        # random location, we need to make sure it falls in the field of play
        for x in range(board.width):
            best_x = randint(0, board.width - i)
            try:
                y = board.drop_height(best_piece, best_x)
                print("IN Play i : {} y: {}".format(i, y))
                break
            except Exception as e:
                print(e)
                # import IPython
                # IPython.embed()
                print("OUT of PLAY")
                i += 1

        print("best_x {} best_piece {}".format(best_x, best_piece))
        return best_x, best_piece
