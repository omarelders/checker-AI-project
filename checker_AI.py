from checkers.game import Game
import numpy as np
import copy
from typing import List
from collections import deque 
from timeit import default_timer as timer

transposition_table = dict()

class full_move:
    def __init__(self, moves=list): 
        self.move_sequence = moves

    def make_move(self, game: Game):
        for move in self.move_sequence:
            game.move(move)

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
        # Generate a unique string representation of the board state
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
        # Create all possible child nodes (next states)
        allBoards = getAllChildBoards(Node(self.game))
        for game, move in allBoards:
            self.children.append(Node(game=game, parent=self, move=move))

    def isQuiet(self) -> bool:
        if self.parent is not None:
            parentPieceCount = sum(1 for piece in self.parent.game.board.pieces if not piece.captured)
            pieceCount = sum(1 for piece in self.game.board.pieces if not piece.captured)
            return pieceCount == parentPieceCount
        return True


def writeBoard(game: Game):
    # Render the board in the terminal
    board = np.full((8,8), '*', dtype=str)
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
            elif piece.player == 1:
                board[y][x] = "w"
            elif piece.player == 2 and piece.king:
                board[y][x] = "B"
            elif piece.player == 2:
                board[y][x] = "b"
    for c in range(8):
        print(*board[c], sep="")
    print()

def getNextBoard(game: Game, move: full_move) -> Game:
    # Create a copy of the board after applying the move
    next_board = copy.deepcopy(game)
    move.make_move(game=next_board)
    return next_board

def getAllChildBoards(node: Node):
    # Generate all possible boards and moves from the current node
    games = []
    legal_moves = getAllFullMoves(node.game)
    for move in legal_moves:
        next_game = getNextBoard(node.game, move)
        games.append((next_game, move))
    return games

def minimaxStart(game: Game, depth: int, maximizingPlayer: bool):
    # Entry point for minimax algorithm
    initialNode = Node(game)
    val = minimax(initialNode, depth, maximizingPlayer)
    transposition_table.clear()
    for child in initialNode.children:
        if child.value == val:
            return child.move

def minimax(node: Node, depth: int, maximizingPlayer: bool):
    # Minimax recursive search
    if depth == 0 or node.isTerminalNode():
        return node.evaluate(maximizingPlayer)

    node.create_children()

    if maximizingPlayer:
        value = -float('inf')
        for child in node.children:
            value = max(value, minimax(child, depth - 1, False))
        node.value = value
        return value
    else:
        value = float('inf')
        for child in node.children:
            value = min(value, minimax(child, depth - 1, True))
        node.value = value
        return value

def getAllFullMoves(game: Game) -> List[full_move]:
    # Retrieve all legal full move sequences
    sequences = game.get_possible_move_sequences()
    return [full_move(seq) for seq in sequences]

def simpleEvaluationFunction(game: Game, maximizingPlayer: bool) -> float:
    # Evaluate board based on number and type of pieces
    white_score = 0
    black_score = 0

    for piece in game.board.pieces:
        if not piece.captured:
            score = 3 if piece.king else 1
            if piece.player == 1:
                white_score += score
            else:
                black_score += score

    return (white_score - black_score) if maximizingPlayer else (black_score - white_score)

if __name__ == "__main__":
    game = Game()
    depth = 3
    human_player = 1  # 1 = white, 2 = black

    print("CHECKERS - Human vs AI")
    print("Move format: Enter numbers like '0' for the first move in the list\n")

    while not game.is_over():
        writeBoard(game)
        
        if game.whose_turn() == human_player:
            legal_moves = getAllFullMoves(game)
            if not legal_moves:
                print("No legal moves available!")
                break
            print("\nYour legal moves:")
            for i, move_obj in enumerate(legal_moves):
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
            print("\nAI thinking...")
            start_time = timer()
            ai_move = minimaxStart(game, depth, maximizingPlayer=(human_player==2))
            ai_move.make_move(game)
            print(f"AI moved positions: {ai_move.move_sequence} (in {timer()-start_time:.1f}s)\n")

    writeBoard(game)
    winner = game.get_winner()
    print("\nGame Over!")
    print("Winner:", "Human" if winner==human_player else "AI" if winner else "Draw")
