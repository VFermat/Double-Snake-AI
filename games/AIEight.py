"""
## AI Model Two
## Used parent move function (see game.py for more info)
## Overrides getState function. State is now the concatenation of the board representation and booleans representing the food position
"""


from .game import Game
import numpy as np
from utils.directions import Directions


class AIEight(Game):
    def __init__(self, size: int, player: int, drawUi: bool = True):
        super().__init__(size, player, drawUi)

    def move(self, action):

        # Moving snake acording to action
        headX, headY = self._takeAction(action)
        self.snake.insert(0, (headX, headY))
        self._fillBoard()

        # Checking reward
        endGame = False
        reward = 0
        if self._checkEndGame():
            reward = -10
            endGame = True

        if self.food == self.snake[0]:
            self.score += 1
            reward = 10
            self.food = self._placeFood()
        else:
            nDist = self._calcDist()
            if nDist > self.dist:
                reward = -5
            else:
                reward = 5
            self.dist = nDist
            self.snake.pop()

        # Increase iterations
        self.iteration += 1
        if self.drawUi:
            self.gameUi.updateUi(self.snake, self.food, self.score)

        return reward, endGame, self.score

    def getState(self):
        head = self.snake[0]
        dir_l = self.direction == Directions.LEFT
        dir_r = self.direction == Directions.RIGHT
        dir_u = self.direction == Directions.UP
        dir_d = self.direction == Directions.DOWN

        possibleCollisions = [dir_l, dir_r, dir_u, dir_d]
        for i in range(1, 5):
            point_l = (head[0] - i, head[1])
            point_r = (head[0] + i, head[1])
            point_u = (head[0], head[1] - i)
            point_d = (head[0], head[1] + i)
            point_lu = (head[0] - i, head[1] - 1)
            point_rd = (head[0] + i, head[1] + i)
            point_ru = (head[0] + i, head[1] - i)
            point_ld = (head[0] - i, head[1] + i)

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
                # Danger straight-right
                (dir_d and self._checkCollision(point_ld))
                or (dir_u and self._checkCollision(point_ru))
                or (dir_r and self._checkCollision(point_rd))
                or (dir_l and self._checkCollision(point_lu)),
                # Danger down-right
                (dir_d and self._checkCollision(point_lu))
                or (dir_u and self._checkCollision(point_rd))
                or (dir_r and self._checkCollision(point_ru))
                or (dir_l and self._checkCollision(point_ld)),
                # Danger straight-left
                (dir_d and self._checkCollision(point_rd))
                or (dir_u and self._checkCollision(point_lu))
                or (dir_r and self._checkCollision(point_ld))
                or (dir_l and self._checkCollision(point_ru)),
                # Danger down-left
                (dir_d and self._checkCollision(point_ru))
                or (dir_u and self._checkCollision(point_ld))
                or (dir_r and self._checkCollision(point_lu))
                or (dir_l and self._checkCollision(point_rd)),
            ]

        foodLocation = [
            self.food[0] < self.snake[0][0],  # food left
            self.food[0] > self.snake[0][0],  # food right
            self.food[1] < self.snake[0][1],  # food up
            self.food[1] > self.snake[0][1],  # food down
        ]
        state = np.concatenate(
            (np.array(possibleCollisions, dtype=int), np.array(foodLocation, dtype=int)))

        return np.array(state, dtype=int)

    def getInputSize(self):
        return len(self.getState())
