# two-player-games
Library of two-player logical games for WUT students

## Architecture

Two primary base classes of the games are `Game` and `State`.

These two classes have very similar functionalities, with the following differences:
 - a `Game` object contains a `State` object
 - making move on a `Game` object changes its state while making move on a `State` object returns new state without affecting the previous one.
 - a `State` object may have more game-specific functionalities

Common functionalities of `Game` and `State`:
 - getting the list of avaliable moves
 - getting current player
 - checking if a game has finished

## Usage example

```python
from two_player_games.games.morris import SixMensMorris  # or any other game
import random


game = SixMensMorris()

while not game.is_finished():
    moves = game.get_moves()
    move = random.choice(moves)
    game.make_move(move)

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)

```
## Detailed usage

### `State` class

`State` objects represent the state of the game at a specific moment. They are treated as immutable - they should not be modified, and will not be modified by a game. They provide access to useful information:

- `State.get_moves(self) -> Iterable[Move]` - provides an iterable object of moves avaliable to current player in current state. Each move returned by this method should be valid.
- `State.get_current_player(self) -> Player` - returns the object that represents current player.
- `State.make_move(self, move: Move) -> State` - performs the selected move in the current state. It does not modify the current state - instead it returns the next state as a separate object.
- `State.is_finished(self) -> bool` - returns `True` if the game is finished in the current state and `False` otherwise.
- `State.get_winner(self) -> Optional[Player]` - return the object that represents the player that is the winner if the game is finished. If the game is not yet finished or there is a draw, it returns `None`.
- `State.get_players(self) -> Iterable[Player]` - return an iterable object of players in the game.
- `State.__str__(self) -> str` - returns a textual representation of the current state in the game.

### `Game` class

`Game` object is a wrapper for a `State` object. A game, when created, creates an initial state. When a move is selected, the game changes its state to the next one.

It provides the same functionality as the `State` object specified above with the following exceptions:

- `Game.make_move(self, move: Move) -> None` - performs the selected move in the current state. It modifies the current state of the game, as opposed to the `State.make_move` method.
- `Game.state` - current state of the game represented by a `State` object.

### `Player` and `Move` classes

`Player` objects are used only to identify players. `Move` objects contain informations about a move and any functionality/data they may provide is strictly game-specific.

