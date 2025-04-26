
from checkers.game import Game
import numpy as np
import copy
from typing import List
from collections import deque 
from timeit import default_timer as timer


transposition_table = dict()

# Definition of the full_move class, which holds a series of moves
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

# Definition of the Node class, which holds all game state information at a position as well as information about its parent and child nodes
class Node:

  # Constructor for Node
  def __init__(self, game : Game, parent = None, move = None):
    self.game = game
    self.children = list()
    self.parent = parent
    self.move = move
    self.value = None
    self.hash = self.hashGame()

  # Function to translate a game state into a string. This is only used with the transposition table improvement.
  def hashGame(self):
    hashed = np.zeros(32, dtype=int) 

    # For each piece, mark its position with an integer that defines it completely
    for piece in self.game.board.pieces:
      if not piece.captured:
        pos = piece.position - 1

        # White kings are denoted with threes
        if piece.player == 1 and piece.king:
          hashed[pos] = 3

        # White pawns are denoted with ones
        elif piece.player == 1 and not piece.king:
          hashed[pos] = 1

        # Black kings are denoted with fours
        elif piece.player == 2 and piece.king:
          hashed[pos] = 4

        # Black pawns are denoted with twos
        elif piece.player == 2 and not piece.king:
          hashed[pos] = 2

    hashedString = ''.join(map(str, hashed))
    return hashedString

  # Function to evaluate a node using an evaluation function
  def evaluate(self, maximizingPlayer):

    # Calculate a value using the simple evaluation function
    self.value = simpleEvaluationFunction(self.game, maximizingPlayer)

    # Record value for use in the transposition table later on
    transposition_table[self.hash] = self.value 

    return self.value

  # Function to determine if there are no more nodes to evaluate beyond this one
  def isTerminalNode(self):
    return self.game.is_over()

  # Function to create all the child nodes of this node
  def create_children(self):

    # Get all of the boards that resulted from moves from the current node
    allBoards = getAllChildBoards(Node(self.game))

    # Add boards as child nodes
    for game, move in allBoards:
      self.children.append(Node(game=game, parent=self, move=move))

  # Function to determine if the node should continue to be expanded when using quiescence search. In this implementation, a node should be expanded if it includes a capture
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


# Function to get the resulting board after a move is made from the current board
def getNextBoard(game : Game, move : full_move) -> Game:
  next_board = copy.deepcopy(game)
  move.make_move(game=next_board)
  return next_board

# Function to get all possible sequences of partial moves from the current board
def getAllFullMoveSequences(game : Game): 
  fullMoveSequences = []
  player_turn = game.whose_turn()

  # Create a deque that is used to hold on to game states at partial moves
  sequenceQueue = deque()
  
  # Add an initial element corresponding to the current game
  current_game = game
  startingEntry = (current_game, [])
  sequenceQueue.append(startingEntry)

  # While the deque is not empty, continue processing move sequences
  while sequenceQueue:
    current_game, sequence = sequenceQueue.pop()

    # If the current element corresponds to a position where the current player can still move, get all possible partial moves from that position
    if current_game.whose_turn() == player_turn:
      moves = current_game.get_possible_moves()

      # For each partial move, create a new sequenceQueue element using the current sequence of moves
      for partial_move in moves:
        next_board = copy.deepcopy(current_game)
        next_board.move(partial_move)
        next_sequence = copy.deepcopy(sequence)
        next_sequence.append(partial_move)
        entry = next_board, next_sequence
        sequenceQueue.append(entry)

    # Otherwise, add the finished move sequence to the overall list
    else:
      fullMoveSequences.append(sequence)
      
  return fullMoveSequences

# Function to get all possible full_moves from the current board
def getAllFullMoves(game : Game)-> List[full_move]:
  full_moves = list()

  # Get every move sequence from the current board
  move_sequences = getAllFullMoveSequences(game)

  # Convert each move sequence to a full_move
  for move_sequence in move_sequences:
    full_moves.append(full_move(move_sequence))

  return full_moves

# Function to get all possible boards that result from moves from the current board
def getAllChildBoards(node : Node) -> List[Game]:
  child_boards = list()

  # Get all possible full_moves from the current board
  all_moves = getAllFullMoves(node.game)

  # For each full_move, get the resulting board
  for move in all_moves:
    child_boards.append((getNextBoard(node.game, move), move))

  return child_boards

# Function to write a text representation of the map to the output
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


# Function which takes in the game state and returns a value describing who is winning
def simpleEvaluationFunction(game : Game, maximizingPlayer : bool):
  turn = game.whose_turn()
  score = 0

  # If the game is over, override the function. We don't need to predict who is winning because the game is decided!
  if game.is_over():

    # In this implementation, we want the bot to avoid stalemates if possible
    if game.get_winner() is None: 
      if maximizingPlayer:
        return -2
      else:
        return 2

    # If white wins, return an extremely high positive value
    elif game.get_winner() == 1:
      return 9999

    # If black wins, return an extremely low negative value
    else:
      return -9999

  # Otherwise, go through each non-captured piece and calculate the simple evaluation function
  for piece in game.board.pieces:
    if not piece.captured:
      if piece.player == 1 and piece.king:
        score += 1.4
      elif piece.player == 1 and not piece.king:
        score += 1
      elif piece.player == 2 and piece.king:
        score -= 1.4
      elif piece.player == 2 and not piece.king:
        score -= 1
        
  return score



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
