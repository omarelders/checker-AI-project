
from checkers.game import Game
import numpy as np
import copy
from typing import List
from collections import deque 
from timeit import default_timer as timer


transposition_table = dict()

class full_move:

  # Constructor for full_move
  def __init__(self, moves = list): 
    self.move_sequence = moves

  # Function to make the sequence of moves in full_move
  def make_move(self, game : Game):
    for move in self.move_sequence:
      game.move(move)

  # Function to compare different full_moves
  def equals(self, full_move2):
    return self.move_sequence == full_move2.move_sequence
  

class Node:
    def __init__(self, game: Game, parent=None, move=None):
        self.game = game
        self.children = []
        self.parent = parent
        self.move = move
        self.value = None
        self.hash = self.hashGame()

    def hashGame(self) -> str:
        hashed = np.zeros(32, dtype=int)
        for piece in self.game.board.pieces:
            if not piece.captured:
                pos = piece.position - 1
                if piece.player == 1 and piece.king:
                    hashed[pos] = 3
                elif piece.player == 1:
                    hashed[pos] = 1
                elif piece.player == 2 and piece.king:
                    hashed[pos] = 4
                elif piece.player == 2:
                    hashed[pos] = 2
        hashedString = ''.join(map(str, hashed))
        return hashedString

    def evaluate(self, maximizingPlayer: bool) -> float:
        self.value = simpleEvaluationFunction(self.game, maximizingPlayer)
        transposition_table[self.hash] = self.value
        return self.value

    def isTerminalNode(self) -> bool:
        return self.game.is_over()

    def create_children(self):
        allBoards = getAllChildBoards(Node(self.game))
        for game, move in allBoards:
            self.children.append(Node(game=game, parent=self, move=move))

    def isQuiet(self) -> bool:
        if self.parent is not None:
            parentPieceCount = sum(1 for piece in self.parent.game.board.pieces if not piece.captured)
            pieceCount = sum(1 for piece in self.game.board.pieces if not piece.captured)
            return pieceCount == parentPieceCount
        return True



class Node:

  def __init__(self, game : Game, parent = None, move = None):
    self.game = game
    self.children = list()
    self.parent = parent
    self.move = move
    self.value = None
    self.hash = self.hashGame()

  def hashGame(self):
    hashed = np.zeros(32, dtype=int) 

    for piece in self.game.board.pieces:
      if not piece.captured:
        pos = piece.position - 1


        if piece.player == 1 and piece.king:
          hashed[pos] = 3

  
        elif piece.player == 1 and not piece.king:
          hashed[pos] = 1

  
        elif piece.player == 2 and piece.king:
          hashed[pos] = 4

  
        elif piece.player == 2 and not piece.king:
          hashed[pos] = 2

    hashedString = ''.join(map(str, hashed))
    return hashedString

  
  def evaluate(self, maximizingPlayer):

  
    self.value = simpleEvaluationFunction(self.game, maximizingPlayer)

  
    transposition_table[self.hash] = self.value 

    return self.value

  
  def isTerminalNode(self):
    return self.game.is_over()

  
  def create_children(self):

  
    allBoards = getAllChildBoards(Node(self.game))

  
    for game, move in allBoards:
      self.children.append(Node(game=game, parent=self, move=move))

  
  
  def isQuiet(self):

    # If this is the root node, it is definitely quiet
    if not self.parent is None:

      # Count the number of pieces in the parent board. If the current board has a different number of pieces, a capture occurred
      parentPieceCount = 0
      for piece in self.parent.game.board.pieces:
        if not piece.captured:
          parentPieceCount += 1
      pieceCount = 0
      for piece in self.game.board.pieces:
        if not piece.captured:
          pieceCount += 1
      if pieceCount == parentPieceCount:
        return True
      else:
        return False

        
    return True


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

<<<<<<< HEAD
def getNextBoard(game : Game, move : full_move) -> Game:
  next_board = copy.deepcopy(game)
  move.make_move(game=next_board)
  return next_board
=======



def simpleEvaluationFunction(game: Game, maximizingPlayer: bool) -> float:
    score = 0
    if game.is_over():
        if game.get_winner() is None:
            return -2 if maximizingPlayer else 2
        elif game.get_winner() == 1:
            return 9999
        else:
            return -9999

    for piece in game.board.pieces:
        if not piece.captured:
            if piece.player == 1 and piece.king:
                score += 1.4
            elif piece.player == 1:
                score += 1
            elif piece.player == 2 and piece.king:
                score -= 1.4
            elif piece.player == 2:
                score -= 1
    return score
>>>>>>> 52e9fab3371113c7d32328afcea499673a6fe669
