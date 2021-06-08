import pygame


class GameUi:
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)

    BLOCK_SIZE = 20

    CLOCK = 40

    def __init__(self, size):
        self.initPygame()
        self.display = pygame.display.set_mode(
            (size * self.BLOCK_SIZE, size * self.BLOCK_SIZE))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

    def initPygame(self):
        pygame.init()
        self.font = pygame.font.Font("arial.ttf", 25)

    def updateUi(self, snake, food, score):
        self.display.fill(self.BLACK)
        for point in snake:
            pygame.draw.rect(self.display, self.BLUE1, pygame.Rect(
                point[0] * self.BLOCK_SIZE, point[1] * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE))

        pygame.draw.rect(self.display, self.RED, pygame.Rect(
            food[0] * self.BLOCK_SIZE, food[1] * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE))
        text = self.font.render("Score: " + str(score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()
