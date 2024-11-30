from two_player_games.games.nim import Nim, NimMove, NimState


def heuristic(s: NimState) -> int:
    return 1


def evaluate(s: NimState) -> int:
    return heuristic(s)


def AlphaBeta(s: NimState, depth: int, max_move: bool, alpha: int, beta: int):
    if s.is_finished or depth == 0:
        return evaluate(s)

    moves = s.get_moves()
    if max_move:
        for move in moves:
            alpha = max(alpha, AlphaBeta(move, depth - 1, not max_move))
            if alpha > beta:
                return alpha
        return alpha
    else:
        for move in moves:
            beta = min(beta, AlphaBeta(move, depth - 1, not max_move))
            if alpha > beta:
                return beta
        return beta
