
from checkers.game import Game

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
