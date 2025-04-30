
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

def getNextBoard(game : Game, move : full_move) -> Game:
  next_board = copy.deepcopy(game)
  move.make_move(game=next_board)
  return next_board




# Function which starts minimax at the current board to the desired depth and backtracks to get the best move
def minimaxStart(game : Game, depth : int, maximizingPlayer : bool):

  # Define the root node at the current board
  initialNode = Node(game)

  # Run minimax and store the resulting value of the root
  val = minimax(initialNode, depth, maximizingPlayer)

  # Backtrack to get the best move (which shares a val with the root)
  currentNode = initialNode
  transposition_table.clear()
  while True:
    for child in currentNode.children:
      if child.value == val:
        return child.move

# The recursive minimax function
def minimax(node : Node, depth : int, maximizingPlayer : bool):

  # If we've reached a terminal node or we've reached the search depth, evaluate the current node
  if depth == 0 or node.isTerminalNode():
    return node.evaluate(maximizingPlayer)

  # Otherwise, expand the node
  node.create_children()

  # If we're currently the maximizing player, get the maximum value of the child nodes
  if maximizingPlayer:
    value = -99999
    for child in node.children:
      value = max([value, minimax(child, depth - 1, False)]) # Get the value of the child node via recursion. Reduce depth to continue searching by 1 and swap maximizingPlayer
    node.value = value
    return value

  # If we're currently the minimizing player, get the minimum value of the child nodes
  else:
    value = 99999
    for child in node.children:
      value = min([value, minimax(child, depth - 1, True)]) # Get the value of the child node via recursion. Reduce depth to continue searching by 1 and swap maximizingPlayer
    node.value = value
    return value


if __name__ == "__main__":
    game = Game()
    depth = 3
    human_player = 1  # 1 = white, 2 = black

    print("CHECKERS - Human vs AI")
    print("Move format: Enter numbers like '0' for the first move in the list\n")

    while not game.is_over():
        writeBoard(game)
        
        if game.whose_turn() == human_player:
            # Human turn - fixed format
            legal_moves = getAllFullMoves(game)
            
            if not legal_moves:
                print("No legal moves available!")
                break
                
            print("\nYour legal moves:")
            for i, move_obj in enumerate(legal_moves):
                # Directly access the position numbers stored in move_sequence
                print(f"{i}: Positions {move_obj.move_sequence}")

            while True:
                try:
                    choice = int(input("Enter move number: "))
                    if 0 <= choice < len(legal_moves):
                        legal_moves[choice].make_move(game)
                        break
                    else:
                        print(f"Invalid number. Enter 0-{len(legal_moves)-1}")
                except ValueError:
                    print("Numbers only please!")
        
        else:
            # AI turn
            print("\nAI thinking...")
            start_time = timer()
            ai_move = minimaxStart(game, depth, maximizingPlayer=(human_player==2))
            ai_move.make_move(game)
            print(f"AI moved positions: {ai_move.move_sequence} (in {timer()-start_time:.1f}s)\n")

    # Game over
    writeBoard(game)
    winner = game.get_winner()
    print("\nGame Over!")
    print("Winner:","Human" if winner==human_player else "AI" if winner else "Draw")
