from .game import Game


class AIOne(Game):
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
            self.snake.pop()

        # Increase iterations
        self.iteration += 1
        if self.drawUi:
            self.gameUi.updateUi(self.snake, self.food, self.score)

        return reward, endGame, self.score
