import pygame
import random


# Constants
WIDTH = 1400
HEIGHT = 1000
PADDLE_SPEED = 20
WINNING_SCORE = 7
FPS = 60

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)


class Paddle:
    def __init__(self, side, window):
        self.window = window
        self.length = HEIGHT // 8
        self.width = WIDTH // 75
        self.color = WHITE
        if side == 'left':
            self.x = self.width
        else:
            self.x = WIDTH - self.width * 2
        self.y = HEIGHT // 2 - self.length // 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.length)


    def draw(self):
        self.rect = pygame.Rect(self.x, self.y, self.width, self.length)
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.width, self.length))


class Ball:
    def __init__(self, window):
        self.window = window
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.x_velocity = random.randint(10, 15) * random.choice([1, -1])
        self.y_velocity = random.randint(-5, 5)
        self.size = 10
        self.color = WHITE
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)


    def update(self):
        self.x += self.x_velocity
        self.y += self.y_velocity
        if self.y < 0 or self.y > HEIGHT - self.size:
            self.y_velocity *= -1

        if self.x < 0:
            return 'right'
        if self.x > WIDTH:
            return 'left'

        return False


    def draw(self):
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.size, self.size))


    def check_bounce(self, paddle):
        if pygame.Rect.colliderect(self.rect, paddle.rect):
            self.x_velocity *= -1
            if paddle.x < WIDTH // 2:
                self.x = paddle.x + paddle.width + 1
                self.x_velocity += 1
            else:
                self.x = paddle .x - self.size - 1
                self.x_velocity -= 1

            self.update_y_velocity(paddle)


    def update_y_velocity(self, paddle):
        y_diff = (self.y + self.size // 2) - (paddle.y + paddle.length // 2)
        self.y_velocity += y_diff // 6


class PongGame:
    def __init__(self, window):
        self.score_left = self.score_right = 0
        self.window = window
        self.initialize_round()


    def initialize_round(self):
        self.left_paddle = Paddle('left', self.window)
        self.right_paddle = Paddle('right', self.window)
        self.ball = Ball(self.window)
        self.draw()
        if max(self.score_left, self.score_right) == WINNING_SCORE - 1:
            announcement_text = FONT2.render(f'Match point!', 1, GREEN)
        else:
            announcement_text = FONT2.render(f'Round begins in...', 1, GREEN)
        self.window.blit(announcement_text, (WIDTH // 2 - announcement_text.get_width() // 2, HEIGHT // 4 - announcement_text.get_height() // 2))

        pygame.display.update()
        pygame.time.wait(1000)
        for countdown in range(3, 0, -1):
            self.draw()
            announcement_text = FONT2.render(f'{countdown}...', 1, GREEN)
            self.window.blit(announcement_text, (WIDTH // 2 - announcement_text.get_width() // 2, HEIGHT // 4 - announcement_text.get_height() // 2))
            pygame.display.update()
            pygame.time.wait(1000)
            pygame.event.clear()

    
    def update(self):
        self.computer_move()
        score = self.ball.update()
        if not score:
            self.ball.check_bounce(self.left_paddle)
            self.ball.check_bounce(self.right_paddle)
            return

        elif score == 'right':
            self.score_right += 1
        elif score == 'left':
            self.score_left += 1
        
        pygame.time.wait(500)
        if max(self.score_left, self.score_right) >= WINNING_SCORE:
            self.endgame()

        self.initialize_round()


    def draw_scores(self):
        score_left_text = FONT.render(f'Computer: {self.score_left}', 1, GREEN)
        self.window.blit(score_left_text, (10, 20 - score_left_text.get_height() // 2))
        score_right_text = FONT.render(f'Player: {self.score_right}', 1, GREEN)
        self.window.blit(score_right_text, (WIDTH - score_right_text.get_width() - 10, 20 - score_right_text.get_height() // 2))


    def draw(self):
        self.window.fill(BLACK)
        self.left_paddle.draw()
        self.right_paddle.draw()
        self.ball.draw()
        self.draw_scores()
        pygame.display.update()


    def player_up(self):
        if self.right_paddle.y > 0:
            self.right_paddle.y -= PADDLE_SPEED


    def player_down(self):
        if self.right_paddle.y < HEIGHT - self.right_paddle.length:
            self.right_paddle.y += PADDLE_SPEED


    def computer_move(self):
        if self.ball.y > self.left_paddle.y + self.left_paddle.length // 2 + PADDLE_SPEED * 2 and self.left_paddle.y < HEIGHT - self.left_paddle.length:
            self.left_paddle.y += PADDLE_SPEED
        elif self.ball.y < self.left_paddle.y + self.left_paddle.length // 2 - PADDLE_SPEED * 2 and self.left_paddle.y > 0:
            self.left_paddle.y -= PADDLE_SPEED

    
    def endgame(self):
        self.draw()
        winner = 'Human' if self.score_right >= 7 else 'Computer'
        greeting_text = FONT.render(f'The {winner} wins!', 1, GREEN)
        self.window.blit(greeting_text, (WIDTH // 2 - greeting_text.get_width() // 2, HEIGHT // 4 - greeting_text.get_height() // 2))
        instructions_text = FONT.render('Press C to play again, or press Q to quit.', 1, GREEN)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 4 + 25 - instructions_text.get_height() // 2))
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


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    global FONT
    FONT = pygame.font.SysFont('verdana', 20, bold=True)
    global FONT2
    FONT2 = pygame.font.SysFont('verdana', 30, bold=True)

    return window


def main():
    window = initialize_pygame()
    pong = PongGame(window)
    clock = pygame.time.Clock()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            main()

        if keys[pygame.K_UP]:
            pong.player_up()

        if keys[pygame.K_DOWN]:
            pong.player_down()

        pong.update()
        pong.draw()


if __name__ == "__main__":
    main()