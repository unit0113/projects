import pygame
pygame.init()
from enum import Enum
from collections import namedtuple
import random
import time


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


WIDTH, HEIGHT = 1000, 1000
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
FONT = pygame.font.SysFont('verdana', 20, bold=True)
BLOCK_SIZE = 20
SPEED = 20

# Colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)


Point = namedtuple('Point', 'x, y')


class Snek:
    def __init__(self, window):
        self.window = window
        self.direction = Direction.RIGHT
        self.head = Point(WIDTH/2, HEIGHT/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.color = WHITE
        self.score = 0
        self._place_food()


    def move(self):
        if self.direction == Direction.RIGHT:
            self.head = Point(self.head.x + BLOCK_SIZE, self.head.y)
        elif self.direction == Direction.LEFT:
            self.head = Point(self.head.x - BLOCK_SIZE, self.head.y)
        elif self.direction == Direction.UP:
            self.head = Point(self.head.x, self.head.y - BLOCK_SIZE)
        elif self.direction == Direction.DOWN:
            self.head = Point(self.head.x, self.head.y + BLOCK_SIZE)

        self.snake.insert(0, self.head)
        if self.head != self.food:
            self.snake.pop()
        else:
            self._place_food()
            self.score += 1

        if self._is_dead():
            endgame(self.score)

    
    def turn(self, direction):
        self.draw()
        self.direction = direction


    def _place_food(self):
        x = random.randint(0, (WIDTH-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (HEIGHT-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def _is_dead(self):
        # Check collision
        if self.snake.count(self.head) > 1:
            return True

        # Check out of bounds
        if (self.head.x < 0 or
            self.head.x > WIDTH - BLOCK_SIZE or
            self.head.y < 0 or
            self.head.y > HEIGHT - BLOCK_SIZE
            ):
            return True

        return False


    def draw(self):
        self.window.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.window, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.window, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.window, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = FONT.render("Score: " + str(self.score), True, WHITE)
        self.window.blit(text, [0, 0])
        pygame.display.flip()



def endgame(score, window=WINDOW):
    time.sleep(1)
    window.fill(BLACK)
    greeting_text = FONT.render('Game Over', 1, GREEN)
    window.blit(greeting_text, (WIDTH // 2 - greeting_text.get_width() // 2, HEIGHT // 2 - 30 - greeting_text.get_height() // 2))
    greeting_text_1 = FONT.render(f'You final score is {score}', 1, GREEN)
    window.blit(greeting_text_1, (WIDTH // 2 - greeting_text_1.get_width() // 2, HEIGHT // 2 - greeting_text_1.get_height() // 2))
    instructions_text_1 = FONT.render('Press C to play again', 1, GREEN)
    window.blit(instructions_text_1, (WIDTH // 2 - instructions_text_1.get_width() // 2, HEIGHT // 2 + 30 - instructions_text_1.get_height() // 2))
    instructions_text_2 = FONT.render('Or press Q to quit.', 1, GREEN)
    window.blit(instructions_text_2, (WIDTH // 2 - instructions_text_2.get_width() // 2, HEIGHT // 2 + 60 - instructions_text_2.get_height() // 2))
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                main()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                quit()


def main():
    clock = pygame.time.Clock()
    snek = Snek(WINDOW)

    while True:
        clock.tick(SPEED)
        snek.draw()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Turn
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and snek.direction != Direction.RIGHT:
                    snek.turn(Direction.LEFT)
                elif event.key == pygame.K_RIGHT and snek.direction != Direction.LEFT:
                    snek.turn(Direction.RIGHT)
                elif event.key == pygame.K_UP and snek.direction != Direction.DOWN:
                    snek.turn(Direction.UP)
                elif event.key == pygame.K_DOWN and snek.direction != Direction.UP:
                    snek.turn(Direction.DOWN)


        snek.move()


if __name__ == "__main__":
    main()
