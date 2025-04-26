
from checkers.game import Game
import numpy as np
import copy
from typing import List
from collections import deque 
from timeit import default_timer as timer


transposition_table = dict()


def writeBoard(game : Game):
  board = np.full((8,8), '*', dtype = str)
  for piece in game.board.pieces:
    if not piece.captured:
      pos = piece.position
      y = int((pos - 1) / 4)  
      if y % 2 == 0:
        x = 2 * (pos - (4 * y)) - 1
      else:
        x = 2 * (pos - (4 * y)) - 2
      if piece.player == 1 and piece.king:
        board[y][x] = "W"
      elif piece.player == 1 and not piece.king:
        board[y][x] = "w"
      elif piece.player == 2 and piece.king:
        board[y][x] = "B"
      elif piece.player == 2 and not piece.king:
        board[y][x] = "b"
  for c in range(8):
    print(*board[c], sep="")
  print()

