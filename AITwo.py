"""
## AI Model Two
## Used parent move function (see game.py for more info)
## Overrides getState function. State is now the concatenation of the board representation and booleans representing the food position
"""


from game import Game
import numpy as np


class AITwo(Game):
    def __init__(self, size: int, player: int, drawUi: bool = True):
        super().__init__(size, player, drawUi)

    def getState(self):
        head = self.snake[0]
        dir_l = self.direction == Directions.LEFT
        dir_r = self.direction == Directions.RIGHT
        dir_u = self.direction == Directions.UP
        dir_d = self.direction == Directions.DOWN

        possibleCollisions = [dir_l, dir_r, dir_u, dir_d]
        for i in range(1, 6):
            point_l = (head[0] - i, head[1])
            point_r = (head[0] + i, head[1])
            point_u = (head[0], head[1] - i)
            point_d = (head[0], head[1] + i)

            possibleCollisions += [
                # Danger straight
                (dir_r and self._checkCollision(point_r))
                or (dir_l and self._checkCollision(point_l))
                or (dir_u and self._checkCollision(point_u))
                or (dir_d and self._checkCollision(point_d)),
                # Danger right
                (dir_u and self._checkCollision(point_r))
                or (dir_d and self._checkCollision(point_l))
                or (dir_l and self._checkCollision(point_u))
                or (dir_r and self._checkCollision(point_d)),
                # Danger left
                (dir_d and self._checkCollision(point_r))
                or (dir_u and self._checkCollision(point_l))
                or (dir_r and self._checkCollision(point_u))
                or (dir_l and self._checkCollision(point_d)),
            ]

        foodLocation = [
            self.food[0] < self.snake[0][0],  # food left
            self.food[0] > self.snake[0][0],  # food right
            self.food[1] < self.snake[0][1],  # food up
            self.food[1] > self.snake[0][1],  # food down
        ]
        state = np.concatenate((np.array(possibleCollisions, dtype=int), np.array(foodLocation, dtype=int)))

        return np.array(state, dtype=int)

    def getInputSize(self):
        return len(self.getState())
