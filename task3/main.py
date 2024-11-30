from two_player_games.games.nim import Nim
from nim_player import NimPlayer
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np


def playGame(depth1: int, depth2: int, heaps: Iterable[int] = (7, 7, 7)) -> str:
    player1 = NimPlayer(depth1)
    player2 = NimPlayer(depth2)
    game = Nim(heaps=heaps, first_player=player1, second_player=player2)

    while not game.is_finished():
        current_player = game.get_current_player()
        move = current_player.get_best_move(game.state)
        game.make_move(move)
    return player1 == game.get_winner()


def get_plot(data, heaps):
    nimsum = 0
    for heap in heaps:
        nimsum ^= heap

    data_sliced = data[1:, 1:]

    plt.imshow(data_sliced, cmap="viridis", interpolation="none")
    plt.colorbar(ticks=[0, 1], label="Player won")  # Show color scale with labels
    plt.xlabel("Depth 1")
    plt.ylabel("Depth 2")
    plt.title(
        "1(yellow) - first player won\n0(purple)-second player won\nnimsum="
        + str(nimsum)
    )

    plt.gca().invert_yaxis()

    plt.xticks(ticks=np.arange(5), labels=np.arange(1, 6))
    plt.yticks(ticks=np.arange(5), labels=np.arange(1, 6))

    plt.show()


def experiment1():
    print("===== experiment 2 ======")
    data = np.zeros((6, 6), dtype=int)
    heaps = (7, 7, 7)

    for i in range(1, 6):
        for j in range(1, 6):
            player1_won = playGame(i, j, heaps)
            print(
                "player1 depth: "
                + str(i)
                + " player2 depth: "
                + str(j)
                + " player 1 won: "
                + str(player1_won)
            )
            data[i][j] = 1 if player1_won else -1

    get_plot(data, heaps)


def experiment2():
    print("===== experiment 2 ======")
    data = np.zeros((6, 6), dtype=int)
    heaps = (1, 2, 4)
    for i in range(1, 6):
        for j in range(1, 6):
            player1_won = playGame(i, j, heaps)
            data[i][j] = 1 if player1_won else -1
    get_plot(data, heaps)


def experiment3():
    print("===== experiment 3 ======")
    heaps = (1, 3, 5, 7)
    data = np.zeros((6, 6), dtype=int)
    for i in range(1, 6):
        for j in range(1, 6):
            player1_won = playGame(i, j, heaps)
            data[i][j] = 1 if player1_won else -1

    get_plot(data, heaps)


def experiment4():
    print("===== experiment 4 ======")
    data = np.zeros((6, 6), dtype=int)
    heaps = (10, 10, 10, 10, 10)
    for i in range(1, 6):
        for j in range(1, 6):
            player1_won = playGame(i, j, heaps)
            data[i][j] = 1 if player1_won else -1

    get_plot(data, heaps)


def experiment5():
    print("===== experiment 5 ======")
    data = np.zeros((6, 6), dtype=int)
    heaps = (7, 7, 7)

    for i in range(1, 6):
        for j in range(1, 6):
            for _ in range(20):
                player1_won = playGame(i, j, heaps)
                print(
                    "player1 depth: "
                    + str(i)
                    + " player2 depth: "
                    + str(j)
                    + " player 1 won: "
                    + str(player1_won)
                )
                data[i][j] += (1.0 if player1_won else -1.0) // 10.0

    get_plot(data, heaps)


def experiment6():
    print("===== experiment 6 ======")
    data = np.zeros((6, 6), dtype=int)
    heaps = (1, 3, 5, 7)

    for i in range(1, 6):
        for j in range(1, 6):
            for _ in range(20):
                player1_won = playGame(i, j, heaps)
                print(
                    "player1 depth: "
                    + str(i)
                    + " player2 depth: "
                    + str(j)
                    + " player 1 won: "
                    + str(player1_won)
                )
                data[i][j] += (1.0 if player1_won else -1.0) // 10.0

    get_plot(data, heaps)


def main():
    # experiment1()
    # experiment2()
    # experiment3()
    # experiment4()
    experiment5()
    #  experiment6()
    # print(playGame(4, 1))


if __name__ == "__main__":
    main()
