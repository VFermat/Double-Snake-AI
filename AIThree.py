"""
## AI Model Three
## Overrides move function to alter reward construction. Positive reward is now also given when snake gets closer to food, and negative when it moves away
## Overrides getState function. State is now the concatenation of the board representation and booleans representing the food position (same as Model Two)
"""


import numpy as np

from game import Game
from utils.directions import Directions


class AIThree(Game):
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
