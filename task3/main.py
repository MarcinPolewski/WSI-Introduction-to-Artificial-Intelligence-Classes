from two_player_games.games.nim import Nim
from nim_player import NimPlayer


def playGame(depth1: int, depth2: int):
    player1 = NimPlayer(depth1)
    player2 = NimPlayer(depth2)
    game = Nim(first_player=player1, second_player=player2)

    while not game.is_finished():
        current_player = game.get_current_player()
        move = current_player.get_best_move(game.state)
        game.make_move(move)

    return game.get_winner()


def main():
    print(playGame(2, 2))


if __name__ == "__main__":
    main()
