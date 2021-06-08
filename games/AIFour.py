"""
## AI Model Three
## Overrides move function to alter reward construction. Positive reward is now also given when snake gets closer to food, and negative when it moves away (same as model three)
## Uses base getState function.
"""


from .game import Game
import numpy as np


class AIFour(Game):
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
