from two_player_games.player import Player
from two_player_games.games.nim import Nim, NimMove, NimState


class NimPlayer(Player):

    def __init__(self, depth, max_player: bool):
        self.depth = depth
        self.max_player = max_player

    def heuristic(self, s: NimState, multiplier: int):
        heaps = s.heaps
        nim_sum = 0
        for heap in heaps:
            nim_sum ^= heap

        if nim_sum == 0:
            return 50 * multiplier
        else:
            return -50 * multiplier

    def evaluate(self, s: NimState):
        multiplier = 1
        if self.max_player:
            multiplier = -1
        if s.is_finished():
            if s.get_winner() == self:
                return multiplier * 100
            else:
                return multiplier * (-100)
        else:
            return self.heuristic(s, multiplier)

    def alphaBetaFinder(
        self, s: NimState, d: int, max_move: bool, alpha: int, beta: int
    ) -> tuple[NimMove, int]:
        if s.is_finished() or d == 0:
            return None, self.evaluate(s)

        moves = s.get_moves()
        bestMove = None

        if self.max_player:
            maxScore = float("-inf")
            for move in moves:
                _, subTreeScore = self.alphaBetaFinder(
                    s.make_move(move), d - 1, False, alpha, beta
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
                    s.make_move(move), d - 1, True, alpha, beta
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
            s, self.depth, self.max_player, float("-inf"), float("+inf")
        )
        return bestMove
