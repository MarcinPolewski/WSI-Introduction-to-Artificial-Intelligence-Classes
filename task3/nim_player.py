from two_player_games.player import Player
from two_player_games.games.nim import NimMove, NimState
from random import shuffle


class NimPlayer(Player):

    def __init__(self, depth):
        self.depth = depth

    def heuristic(self, state: NimState, modifier: int):

        nim_sum = 0
        for heap in state.heaps:
            nim_sum ^= heap

        # corner case 1 - on heap left
        if sum(state.heaps) == 1:
            return modifier * 4

        # corner case 2 - heaps of size 1 left
        if all(x <= 1 for x in state.heaps):
            return modifier * (-1 if state.heaps.count(1) % 2 else 3)

        # normal case - to be solved with nimsum
        return modifier * (1 if nim_sum == 0 else -1)

    def evaluate(self, s: NimState, is_max_player: bool):
        multiplier = 1 if is_max_player else -1

        return self.heuristic(s, multiplier)
        # if s.is_finished():
        #     if s.get_winner() == self:  # self is always a max player
        #         return multiplier * 3
        #     else:
        #         return multiplier * (-3)
        # else:
        #     return self.heuristic(s, multiplier)

    def alphaBetaFinder(
        self, s: NimState, d: int, max_move: bool, alpha: int, beta: int
    ) -> tuple[NimMove, int]:
        if s.is_finished() or d == 0:
            return None, self.evaluate(s, max_move)

        moves = s.get_moves()
        shuffle(moves)
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
                if maxScore >= beta:
                    return bestMove, maxScore
            return bestMove, maxScore
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
                if alpha >= minScore:
                    return bestMove, minScore
            return bestMove, minScore

    def get_best_move(self, s: NimState):
        bestMove, score = self.alphaBetaFinder(
            s, self.depth, True, float("-inf"), float("+inf")
        )
        return bestMove
