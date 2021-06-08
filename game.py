import numpy as np
import random
from utils.directions import Directions
from ui import GameUi


class Game:
    def __init__(self, size: int, player: int, drawUi: int):
        self.size = size
        self.drawUi = drawUi
        self.board = np.array([0] * self.size * self.size, dtype=int)
        self.player = player
        if self.drawUi:
            self.gameUi = GameUi(size)
        self.reset()

    def reset(self):
        self.board = np.array([0] * self.size * self.size, dtype=int)
        self._createSnake()
        self.food = self._placeFood()
        self._fillBoard()
        self.direction = Directions.UP
        self.iteration = 0
        self.score = 0
        self.dist = 100
        if self.drawUi:
            self.gameUi.updateUi(self.snake, self.food, self.score)

    def getState(self):
        head = self.snake[0]

        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = self.direction == Directions.LEFT
        dir_r = self.direction == Directions.RIGHT
        dir_u = self.direction == Directions.UP
        dir_d = self.direction == Directions.DOWN

        state = [
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
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            self.food[0] < self.snake[0][0],  # food left
            self.food[0] > self.snake[0][0],  # food right
            self.food[1] < self.snake[0][1],  # food up
            self.food[1] > self.snake[0][1],  # food down
        ]

        return np.array(state, dtype=int)

    def getInputSize(self):
        return 11

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

    def _placeFood(self):
        xPos = random.randint(2, self.size - 2)
        yPos = random.randint(2, self.size - 2)
        while self.board[xPos * self.size + yPos] == 1:
            xPos = random.randint(2, self.size - 2)
            yPos = random.randint(2, self.size - 2)
        self.board[xPos * self.size + yPos] = 2
        return xPos, yPos

    def _initSnake(self):
        xPos = random.randint(5, self.size - 5)
        yPos = random.randint(5, self.size - 5)
        return xPos, yPos

    def _createSnake(self):
        head = self._initSnake()
        direction = random.randint(0, 3)
        if direction == 0:
            self.snake = [
                head,
                (head[0] + 1, head[1]),
                (head[0] + 2, head[1]),
            ]
        elif direction == 1:
            self.snake = [
                head,
                (head[0] - 1, head[1]),
                (head[0] - 2, head[1]),
            ]
        elif direction == 2:
            self.snake = [
                head,
                (head[0], head[1] + 1),
                (head[0], head[1] + 2),
            ]
        elif direction == 3:
            self.snake = [
                head,
                (head[0], head[1] - 1),
                (head[0], head[1] - 2),
            ]

    def _fillBoard(self):
        nBoard = self.board
        for point in self.snake:
            p = point[0] * self.size + point[1]
            if p > 1023:
                p = 1023
            if p < 0:
                p = 0
            nBoard[p] = 1
        nBoard[self.food[0] * self.size + self.food[1]] = 2
        return nBoard

    def _checkCollision(self, point=None):
        if point is None:
            point = self.snake[0]
        if point in self.snake[1:]:
            return True
        if point[0] > self.size - 1 or point[0] < 1 or point[1] > self.size - 1 or point[1] < 1:
            return True
        return False

    def _checkEndGame(self):
        if self.iteration > 100 * len(self.snake) or self._checkCollision():
            return True
        return False

    def _takeAction(self, action):
        clockWise = [Directions.RIGHT, Directions.DOWN, Directions.LEFT, Directions.UP]
        idx = clockWise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            nDir = clockWise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            nIdx = (idx + 1) % 4
            nDir = clockWise[nIdx]
        elif np.array_equal(action, [0, 0, 1]):
            nIdx = (idx - 1) % 4
            nDir = clockWise[nIdx]

        self.direction = nDir
        headX = self.snake[0][0]
        headY = self.snake[0][1]
        if self.direction == Directions.RIGHT:
            headX += 1
        elif self.direction == Directions.LEFT:
            headX -= 1
        elif self.direction == Directions.DOWN:
            headY += 1
        elif self.direction == Directions.UP:
            headY -= 1

        return headX, headY

    def _calcDist(self):
        head = self.snake[0]
        distX = head[0] - self.food[0]
        distY = head[1] - self.food[1]
        return (distX ** 2 + distY ** 2) ** 0.5
