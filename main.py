import pygame


if __name__ == '__main__':

    game = Game(32, 1)
    running = True
    endGame = False
    while running and not endGame:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    reward, endGame, score = game.move([0, 0, 1])
                elif event.key == pygame.K_RIGHT:
                    reward, endGame, score = game.move([0, 1, 0])
                elif event.key == pygame.K_UP:
                    reward, endGame, score = game.move([1, 0, 0])
