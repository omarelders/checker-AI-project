
from checkers.game import Game
import numpy as np
from evaluation import simpleEvaluationFunction
from board_functions import getAllChildBoards
from constants import transposition_table

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
