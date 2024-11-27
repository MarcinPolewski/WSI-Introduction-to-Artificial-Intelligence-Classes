from two_player_games.player import Player
from two_player_games.games.nim import Nim, NimMove, NimState


class NimPlayer(Player):

    def __init__(self, depth):
        self.depth = depth

    def evaluate(self, s: NimState):
        return 0

    def alphaBetaFinder(
        self, s: NimState, d: int, max_move: bool, alpha: int, beta: int
    ) -> tuple[NimMove, int]:
        if s.is_finished() or d == 0:
            return None, self.evaluate(s)

        moves = s.get_moves()
        bestMove = None

        if max_move:
            maxScore = float("-inf")
            for move in moves:
                _, subTreeScore = self.alphaBetaFinder(
                    s.make_move(move), d - 1, not max_move, alpha, beta
                )
                if subTreeScore > maxScore:
                    bestMove = move
                    maxScore = subTreeScore
                alpha = max(alpha, maxScore)
                if alpha > beta:
                    return bestMove, alpha
            return bestMove, alpha
        else:
            minScore = float("inf")
            for move in moves:
                _, subTreeScore = self.alphaBetaFinder(
                    s.make_move(move), d - 1, not max_move, alpha, beta
                )
                if subTreeScore < minScore:
                    minScore = subTreeScore
                    bestMove = move
                beta = min(beta, minScore)
                if alpha > beta:
                    return bestMove, beta
            return bestMove, beta

    def get_best_move(self, s: NimState):
        bestMove, score = self.alphaBetaFinder(
            s, self.depth, True, float("-inf"), float("+inf")
        )
        return bestMove
