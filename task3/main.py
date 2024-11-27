from two_player_games.games.nim import Nim
from nim_player import NimPlayer


def playGame(depth1: int, depth2: int):
    player1 = NimPlayer(depth1, True)
    player2 = NimPlayer(depth2, False)
    game = Nim(first_player=player1, second_player=player2)

    while not game.is_finished():
        current_player = game.get_current_player()
        move = current_player.get_best_move(game.state)
        game.make_move(move)
        # print(current_player, move)
    return player1 == game.get_winner()


def main():
    for i in range(1, 5):
        for j in range(1, 5):
            player1_won = playGame(i, j)
            print(
                "depth1: "
                + str(i)
                + " depth2: "
                + str(j)
                + " player 1 won: "
                + str(player1_won)
            )


if __name__ == "__main__":
    main()
