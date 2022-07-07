import pygame
import numpy as np


# Constants
WANDER_STRENGTH = 1
WIDTH = 3440
HEIGHT = 1440
CELL_SIZE = 10
FPS = 60

# Colors
COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)


class Game:
    def __init__(self, window):
        self.window = window
        self.progress = False
        self.cells = np.zeros((HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE))

    def pause(self):
        self.progress = not self.progress

    def update(self):
        if not self.progress:
            for row, col in np.ndindex(self.cells.shape):
                color = COLOR_ALIVE_NEXT if self.cells[row, col] else COLOR_BG
                pygame.draw.rect(self.window, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
            return

        updated_cells = np.zeros_like(self.cells)
        for row, col in np.ndindex(self.cells.shape):
            cell_status = self.cells[row, col]
            alive_count = np.sum(self.cells[row-1:row+2, col-1:col+2]) - cell_status
            color = COLOR_BG if not cell_status else COLOR_ALIVE_NEXT

            if cell_status:
                if (alive_count < 2 or alive_count > 3):
                    color = COLOR_DIE_NEXT
                elif 2 <= alive_count <= 3:
                    updated_cells[row, col] = 1
                    color = COLOR_ALIVE_NEXT
            
            else:
                if alive_count == 3:
                    updated_cells[row, col] = 1
                    color = COLOR_ALIVE_NEXT

            pygame.draw.rect(self.window, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
        
        self.cells = updated_cells

    def draw(self):
        self.window.fill(COLOR_GRID)
        self.update()
        pygame.display.update()


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Game of Life")
    game = Game(window)

    return game


def main():
    game = initialize_pygame()
    clock = pygame.time.Clock()
    game.draw()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                # Quit
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

                # Restart
                elif event.key == pygame.K_r:
                    main()

                # Pause updating
                elif event.key == pygame.K_SPACE:
                    game.pause()

            # Draw new alives
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                game.cells[pos[1] // CELL_SIZE, pos[0] // CELL_SIZE] = 1
                game.draw()

        game.draw()
        #pygame.time.wait(5)


if __name__ == "__main__":
    main()